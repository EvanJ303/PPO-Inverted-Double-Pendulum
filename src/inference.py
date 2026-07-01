import gymnasium as gym
import time
from agent import PPOAgent

NUM_EPISODES = 10

def main():
    agent = PPOAgent(9,1)

    env = gym.make('InvertedDoublePendulum-v5', render_mode='human')

    with open('./data/latest_checkpoint.txt', 'r') as f:
        checkpoint_path = f.read().strip()

    agent.load_model(checkpoint_path)

    for episode in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        while not done:
            action = agent.step(obs, training=False)
            obs, reward, terminated, truncated, info = env.step(action)
            time.sleep(1/30)
            done = terminated or truncated

    env.close()

if __name__ == '__main__':
    main()