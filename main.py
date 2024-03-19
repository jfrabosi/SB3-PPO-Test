import gymnasium as gym
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import torch as th
import SerialStewartPlatform



num_ssp = 3
# Custom actor (pi) and value function (vf) networks
# of two layers of size 32 each with Relu activation function
# Note: an extra linear layer will be added on top of the pi and the vf nets, respectively


env = gym.make("SerialStewartPlatform/SerialStewartPlatform-v0", num_ssp=num_ssp)#, render_mode="human")
model = PPO("MlpPolicy", env, verbose=0, device="cpu")

replicates = 1
total_timesteps = [None] * replicates# * 9
mean_reward = [None] * replicates# * 9
std_reward = [None] * replicates# * 9
index = 0
taguchi = np.array([[0, 0, 0],
                    [1, 1, 1],
                    [2, 2, 2],
                    [0, 1, 2],
                    [1, 2, 0],
                    [2, 0, 1],
                    [0, 2, 1],
                    [1, 0, 2],
                    [2, 1, 0]])

# # # for i in range(9):
# for n in range(replicates):
#     del model
#     t1 = time.time()
#     print(index)
#     # pi_layers = [num_ssp * 18 * 2 * (1+taguchi[i, 2]) for _ in range(2 + taguchi[i, 1] * 2)]
#     pi_layers = [num_ssp * 18 * 2 * 2 for _ in range(6)]
#     # print(pi_layers)
#     policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=dict(pi=pi_layers, vf=pi_layers))
#     model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs, device="cpu")
#     total_timesteps[index] = 2000000 #* 2**(taguchi[i, 0])
#     print(total_timesteps[index])
#     model.learn(total_timesteps=total_timesteps[index], progress_bar=True)
#     mean_reward[index], std_reward[index] = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
#     print(f"{total_timesteps[index]:<16.0f}{mean_reward[index]:<16.2f}{std_reward[index]:<16.2f}")
#     index += 1
# model.save("ppo_test_2")
#
# # Print the table header
# print(f"{'Total Timesteps':<16}{'Mean Reward':<16}{'STD Reward':<16}")
#
# # Print the table rows
# for timesteps, mean, std in zip(total_timesteps, mean_reward, std_reward):
#     print(f"{timesteps:<16.0f}{mean:<16.2f}{std:<16.2f}")

env = gym.make("SerialStewartPlatform/SerialStewartPlatform-v0", num_ssp=num_ssp, render_mode="human")

# the policy_kwargs are automatically loaded

model = PPO.load("ppo_test_v4", env=env)

# Enjoy trained agent
vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()