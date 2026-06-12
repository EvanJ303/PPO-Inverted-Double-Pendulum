import matplotlib.pyplot as plt
import gymnasium as gym
import os
from collections import deque
from datetime import datetime
from agent import PPOAgent

agent = PPOAgent(9, 1)

env = gym.make('InvertedDoublePendulum-v5')

os.makedirs('./data/models', exist_ok=True)
os.makedirs('./data/plots', exist_ok=True)

NUM_BUFFER_FILLS = 100
EPOCHS_PER_BUFFER_FILL = 10
MAX_RECENT_EPISODES = 50

obs, info = env.reset()
episode_reward = 0.0
episode_rewards = []
avg_rewards = []
episode_count = 0
recent_rewards = deque(maxlen=MAX_RECENT_EPISODES)

for fill_idx in range(NUM_BUFFER_FILLS):
    for _ in range(agent.buffer.buffer.maxlen):
        z, action, log_prob, value = agent.step(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if terminated or truncated:
            next_value = 0.0
        else:
            next_value = agent.get_value(next_obs)

        agent.store_experience(obs, z, value, next_value, reward, log_prob, terminated or truncated)
        
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            episode_count += 1
            recent_rewards.append(episode_reward)
            running_avg = sum(recent_rewards) / len(recent_rewards)
            print(
                f"[Buffer {fill_idx + 1}/{NUM_BUFFER_FILLS}] "
                f"Episode {episode_count} reward={episode_reward:.2f} "
                f"running_avg_last{len(recent_rewards)}={running_avg:.2f}"
            )
            episode_reward = 0.0
            obs, info = env.reset()
        else:
            obs = next_obs

    for _ in range(EPOCHS_PER_BUFFER_FILL):
        agent.optimize_model()

    agent.buffer.clear()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = f'./data/models/ppo_agent_{timestamp}.pth'
    agent.save_model(checkpoint_path)
    
    with open('./data/latest_checkpoint.txt', 'w') as f:
        f.write(checkpoint_path)

    if episode_rewards:
        avg_rewards.append(sum(episode_rewards) / len(episode_rewards))
    else:
        avg_rewards.append(0.0)

    print(
        f"Completed buffer fill {fill_idx + 1}/{NUM_BUFFER_FILLS}: "
        f"total_episodes={episode_count}, overall_avg_reward={avg_rewards[-1]:.2f}"
    )

    fig = plt.figure(figsize=(8, 4))
    plt.plot(avg_rewards, marker='o', linewidth=2)
    plt.xlabel('Buffer fill')
    plt.ylabel('Average episode return')
    plt.title('Training performance')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig('./data/plots/training_rewards.png')
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4))
    plt.plot(avg_rewards, marker='o', linewidth=2)
    plt.xlabel('Buffer fill')
    plt.ylabel('Average episode return')
    plt.title('Training performance')
    plt.grid(True)
    plt.tight_layout()
    fig.savefig(f'./data/plots/training_rewards_{timestamp}.png')
    plt.close(fig)