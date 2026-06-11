import matplotlib.pyplot as plt
import gymnasium as gym
from datetime import datetime
from agent import PPOAgent

agent = PPOAgent(9, 1)

env = gym.make('InvertedDoublePendulum-v5')

for step in range(len(agent.buffer.buffer)):