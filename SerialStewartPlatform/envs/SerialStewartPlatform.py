import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import io
import time
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


class SerialStewartPlatform(gym.Env):
    """
    A class representing a Gymnasium environment for simulating serial Stewart platforms
    as 2*num_ssp vectors.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, num_ssp=3, default_len=0.6, max_dist=0.2, max_angle=20.0):
        """
        initializes the SSP.
        :param render_mode: str; how the SSP should be rendered. choose "None" for no rendering and "human" for
                            rendering within a matplotlib 3D plot.
        :param num_ssp: int; the number of Stewart platforms in the SSP
        :param default_len: float; the "neutral" length of each Stewart platform in meters. if a platform can move its
                            top platform between 0.8m and 0.4m vertically from its base platform, it would have a
                            default_len of 0.6m
        :param max_dist: float; the maximum linear extension (or retraction) of each Stewart platform in meters. in the
                         above example, the platform would have a max_dist of 0.2m
        :param max_angle: float; the maximum angle that each Stewart platform's top plane can reach relative to the its
                          base. if a platform can reach an angle of 30 degrees between the two planes of the platform,
                          it has a max_angle of 30 deg.
        """
        self.num_ssp = num_ssp          # number of serial stewart platforms to simulate
        self.default_len = default_len  # default (middle) length of platform, in m
        self.max_dist = max_dist        # max translation allowed, in m
        self.max_angle = max_angle      # max angle allowed between planes, in deg
        self.timesteps_this_rep = 0     # number of timesteps for current desired position/orientation
        self.positions_hit = 0          # number of times desired position/orientation has been achieved, per simulation
        self._init_hyperparameters()

        # observation space is list of position and orientation vectors that describe SSP
        self.observation_space = spaces.Box(low=-100, high=100, shape=(self.num_ssp*2 + 2, 3), dtype=np.float64)

        # action space is list of two 3x3 rotation matrices to apply to each Stewart platform,
        # resulting in a change in position or orientation for each platform
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.num_ssp, 3, 6), dtype=np.float64)

        # rendering stuff
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Set up the initial figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    
    def _init_hyperparameters(self):
        """
        initializes hyperparameters used for tuning the rewards and simulation of the SSP.
        :return: none
        """
        self._action_scale = 0.01                       # factor to scale actions by (converting to delta_tf's)
        self._base_vector = [0, 0, self.default_len]    # base vector of SSP, always pointing up
        self._min_dist_for_term = 0.1                   # minimum distance for termination (goal reached), m
        self._min_angle_for_term = 5                    # minimum angle for termination (goal reached), deg
        self._nv_weight = 1 / self.max_angle            # scaling factor for normal vector angle in rewards sum (bad)
        self._xyz_weight = 1 / self.default_len         # scaling factor for top platform position in rewards sum (bad)
        self._correct_pos_weight = 100                  # scaling factor for hitting correct (angle+xyz) in rewards sum
        self._overexertion_weight = 50                  # scaling factor for overexerting (bad)
        self._timesteps_weight = 0.001                  # scaling factor for number of timesteps taken (bad)

    def _find_angle(self, base_vector, top_vector):
        """
        finds the current angle of a Stewart platform, i.e. the angle between its base and its top, using:
        angle = arc_cos( a*b / (|a| * |b|) ), or the inverse cosine of the dot product between the vectors
        divided by the product of their magnitude
        :param base_vector: a 3x1 vector representing the normal vector for the base of the platform
        :param top_vector: a 3x1 vector representing the normal vector for the top of the platform
        :return: the angle between the platforms (float)
        """
        # find dot product
        dot_product = np.dot(base_vector, top_vector)

        # find magnitudes and multiply them
        top_mag = self._find_length(top_vector)
        base_mag = self._find_length(base_vector)

        # find angle between vectors, convert to degrees
        angle = np.arccos(dot_product / (top_mag*base_mag)) * 180 / np.pi
        return angle

    def _find_length(self, vector):
        """
        calculates the length (magnitude) of a vector
        :param vector: 3x1 vector to find magnitude of
        :return: magnitude of vector
        """
        length = np.sqrt(sum([vector[i]**2 for i in range(3)]))
        return length

    def _reset_desired_vectors(self):
        # radius of point within sphere of radius num_ssp * max_len * scaling factor
        radius = self.num_ssp * self.max_dist * np.cbrt(np.random.rand()) * 0.8

        # generate random angles
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        phi = np.random.uniform(0, np.pi)  # Polar

        # convert to cartesian and translate z to default position of platform
        x_prime = radius * np.sin(phi) * np.cos(theta)
        y_prime = radius * np.sin(phi) * np.sin(theta)
        z_prime = radius * np.cos(phi) + self.num_ssp * self.default_len

        # initialize desired top platform center XYZ
        self._desired_xyz = np.array([x_prime, y_prime, z_prime])

        # initialize desired top platform normal vector
        self._desired_nv = np.random.uniform(low=[-1.0, -1.0, 0.5], high=[1.0, 1.0, 1.0])

        # normalize desired vector
        nv_mag = self._find_length(self._desired_nv)

        # avoid dividing by zero
        if nv_mag == 0:
            nv_mag = 1

        self._desired_nv = self._desired_nv / nv_mag

        # reset timesteps for this vector
        self.timesteps_this_rep = 0

    def _get_info(self):
        return {"hi": 0}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.positions_hit = 0

        # initialize all orientation and position vectors to be equal to the base vector [0, 0, default_len]
        self.orientation_vectors = np.tile(self._base_vector, (self.num_ssp, 1))
        self.position_vectors = np.tile(self._base_vector, (self.num_ssp, 1))

        # reset desired vectors (desired orientation and position of top platform)
        self._reset_desired_vectors()
        desired_angle = self._find_angle(self._desired_nv, self._base_vector)

        # ensure that angle is not too steep for platform to reach
        while desired_angle > self.max_angle:
            self._reset_desired_vectors()
            desired_angle = self._find_angle(self._desired_nv, self._base_vector)

        # save observation (all position and orientation vectors, and desired position and orientation vector)
        observation = np.vstack(([self.position_vectors, self.orientation_vectors, self._desired_xyz, self._desired_nv]))
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        takes in an action, and simulates movement of the SSP according to that action, returning a reward and an
        observation of the post-movement state
        :param action: a num_ssp by 3 by 4 array containing a 3x3 rotation matrix and 1x3 translation vector to be
                       applied to each num_ssp Stewart platform
        :return: a reward (float) and an observation (2*(num_ssp+1) by 3 array of vectors)
        """
        # reset overexertion and correct position flags, which keep track of whether the agent tried to move a platform
        # beyond its range of motion, and whether the correct position/orientation was achieved in this timestep
        overexertion_flag = 0
        correct_position_flag = 0
        overexertion_count = 0

        # scale action array down by hyperparameter (keeping the action matrices at original bounds produces too
        # large of movement; we want changes in the SSP's orientation and position to be gradual and small
        _action_scaled = self._action_scale * action

        # separate nx3x6 array into two nx3x3 matrices for orientation and position vectors
        delta_orientation_tf = _action_scaled[:, :, 0:3]
        delta_position_tf = _action_scaled[:, :, 3:6]

        # iterate through matrices for each serial platform
        for i in range(self.num_ssp):

            # make a 3x3 identify transformation matrix
            id_tf = np.eye(3)

            # add delta array for current platform to identity matrix
            new_orientation_tf = id_tf + delta_orientation_tf[i, :, :]
            new_position_tf = id_tf + delta_position_tf[i, :, :]

            # the normal vector for the top of the ith platform is the ith entry in the orientation_vector array
            old_orientation_vector = self.orientation_vectors[i, :]

            # transform the old orientation vector using the new transformation
            new_orientation_vector = np.matmul(new_orientation_tf, old_orientation_vector)

            # if this is the first platform, the base vector is just the standard base vector
            if i == 0:
                base_normal_vector = self._base_vector
            # otherwise, the base vector is the (i-1)th entry in the orientation_vector array
            else:
                base_normal_vector = self.orientation_vectors[-1, :]

            # the position vector describing the distance between the platform's base and top is the ith entry in
            # the position_vector array
            old_position_vector = self.position_vectors[i, :]

            # transform the old position vector using the new transformation
            new_position_vector = np.matmul(new_position_tf, old_position_vector)

            # if new transform would make current orientation vector exceed max_angle, do not transform
            if self._find_angle(base_normal_vector, new_orientation_vector) > self.max_angle:
                overexertion_flag = 1
                overexertion_count += 1

            # if new transform would make current position vector exceed max_len, do not transform
            if abs(self._find_length(new_position_vector) - self.default_len) > self.max_dist:
                overexertion_flag = 1
                overexertion_count += 1

            # if new transformation does not overreach bounds of platform, save new vectors in list
            # make sure not to save the 4th dimension of the vector, it's just for multiplication
            if overexertion_flag == 0:
                self.orientation_vectors[i, :] = new_orientation_vector
                self.position_vectors[i, :] = new_position_vector

        # calculate xyz position of the center of each platform
        self.xyz_list = np.zeros([self.num_ssp, 3])
        prev_vector = np.zeros([1, 3])
        for i in range(self.num_ssp):
            self.xyz_list[i, :] = self.position_vectors[i, :] + prev_vector
            prev_vector += self.position_vectors[i, :]

        # return distance between actual top platform XYZ and desired
        xyz_dist = 0
        for i in range(3):
            xyz_dist += (self.xyz_list[self.num_ssp-1, i] - self._desired_xyz[i])**2
        xyz_dist = np.sqrt(xyz_dist)

        # return angle between actual top platform normal vector and desired
        nv_angle = abs(self._find_angle(self._desired_nv, self.orientation_vectors[self.num_ssp-1, :]))

        # check if desired position/orientation has been reached; if so, flip flag, and give new desired position
        if (xyz_dist <= self._min_dist_for_term) and (nv_angle <= self._min_angle_for_term):
            correct_position_flag = 1
            self.positions_hit += 1
            self._reset_desired_vectors()

        # calculate reward: sum of (negative) distance from desired xyz position, distance from desired top orientation,
        #                   number of timesteps so far to reach the desired position/orientation. give a big reward if
        #                   the position is reached, and give a big punishment if the platform would overexert
        reward = -xyz_dist*self._xyz_weight - nv_angle*self._nv_weight - self.timesteps_this_rep*self._timesteps_weight
        reward += self._correct_pos_weight * correct_position_flag
        reward -= self._overexertion_weight * overexertion_flag

        # save observation (all position and orientation vectors, and desired position and orientation vector)
        observation = np.vstack([self.position_vectors, self.orientation_vectors, self._desired_xyz, self._desired_nv])

        # if more than num_ssp overexertions are found or if correct position is hit, terminate
        terminated = bool(overexertion_count >= self.num_ssp or correct_position_flag)
        self._reward = reward
        self._xyz_dist = xyz_dist
        self._nv_angle = nv_angle
        info = self._get_info()

        #print(xyz_dist, nv_angle, self.timesteps_this_rep, reward)
        if correct_position_flag:
            print("hit!")
        # print(observation)
        # print(self.xyz_list)

        if self.render_mode == "human":
            self._render_frame()

        self.timesteps_this_rep += 1

        # if terminated:
        #     self.reset()
        # else:
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        # add desired position to circle centers
        circle_centers = np.vstack((self.xyz_list, self._desired_xyz))

        # add desired nv to normal vectors
        normal_vectors = np.vstack((self.orientation_vectors, self._desired_nv))

        self.plot_circles(circle_centers, normal_vectors, radius=0.4)

    def circle_points(self, radius, center, num_points=100):
        """
        Generate points on a circle's circumference in 2D.
        """
        angles = np.linspace(0, 2 * np.pi, num_points)
        x = center[0] + radius * np.cos(angles)
        y = center[1] + radius * np.sin(angles)
        return np.vstack((x, y)).T  # Shape: num_points x 2

    def rotate_to_normal(self, circle_points, normal_vector):
        """
        Rotate 2D circle points in the xy-plane to align with the given 3D normal vector.
        """
        # Normal vector for the xy-plane circle
        z_axis = np.array([0, 0, 1])
        # Compute the rotation vector via cross product
        rotation_vector = np.cross(z_axis, normal_vector)
        # Compute the rotation angle
        sin_angle = np.linalg.norm(rotation_vector)
        cos_angle = np.dot(z_axis, normal_vector)
        angle = np.arctan2(sin_angle, cos_angle)
        # Normalize the rotation vector
        rotation_vector_normalized = rotation_vector / sin_angle
        # Compute the rotation matrix
        rotation = R.from_rotvec(rotation_vector_normalized * angle)
        # Apply the rotation to the circle points
        circle_points_3d = np.hstack((circle_points, np.zeros((circle_points.shape[0], 1))))  # Add z=0 for all points
        return rotation.apply(circle_points_3d)

    def plot_circles(self, xyz_centers, normals, radius, num_points=100):
        # plot platforms
        for center, normal in zip(xyz_centers, normals):
            # Generate 2D circle points
            circle_2d = self.circle_points(radius, [0, 0], num_points)
            # Rotate to align with the normal vector and translate to the 3D center
            circle_3d = self.rotate_to_normal(circle_2d, normal) + center
            # Plot
            self.ax.plot(circle_3d[:, 0], circle_3d[:, 1], circle_3d[:, 2])

        # plot base
        angles = np.linspace(0, 2 * np.pi, num_points)
        base_x = radius * np.cos(angles)
        base_y = radius * np.sin(angles)
        base_z = np.zeros(num_points)  # All points have the same z coordinate since the circle is in the xy-plane
        self.ax.plot(base_x, base_y, base_z)

        # Plot vectors from one circle center to the next
        for i in range(len(xyz_centers) - 1):  # -1 because the last center doesn't have a next center
            if i == 0:
                start_point = [0, 0, 0]
            else:
                start_point = xyz_centers[i-1]
            end_point = xyz_centers[i]
            # Draw the vector as a line from start_point to end_point
            self.ax.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], [start_point[2], end_point[2]],
                         color='k', linestyle='--')

        max_height = self.num_ssp*(self.default_len+self.max_dist)
        max_width = self.num_ssp*np.sin(self.max_angle)

        # Set plot limits and labels
        self.ax.set_xlim([-max_width, max_width])
        self.ax.set_ylim([-max_width, max_width])
        self.ax.set_zlim([0, max_height])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Serial Stewart Platform Vectors')

        # Check if there's an existing text annotation and remove it
        if hasattr(self, 'text_annotation'):
            self.text_annotation.remove()

        # Add new text annotation
        # Format the string with the desired values
        text_str = "{:.1f}, {:.2f}, {:.1f}, {:.0f}".format(self._reward, self._xyz_dist, self._nv_angle,
                                                           self.timesteps_this_rep)

        # Split the formatted string into individual parts
        reward_str, xyz_dist_str, nv_angle_str, timesteps_str = text_str.split(", ")

        # Use the parts in the text annotation
        self.text_annotation = self.ax.text2D(0.05, 0.95,
                                              f"Reward: {reward_str}, XYZ Diff: {xyz_dist_str}"
                                              f", Angle Diff: {nv_angle_str}, Timesteps: {timesteps_str}",
                                              transform=self.ax.transAxes)

        if self.render_mode == 'human':
            plt.draw()
            plt.pause(0.01)
            plt.cla()

        elif self.render_mode == 'rgb_array':
            # Save the plot to a buffer and return it
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            img = plt.imread(buf)
            buf.close()
            plt.close(self.fig)
            return img