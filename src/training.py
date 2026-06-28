import matplotlib.pyplot as plt
import gymnasium as gym
import os
from collections import deque
from datetime import datetime
from agent import PPOAgent

# Initialize PPO agent: state_dim=9, action_dim=1 (continuous control)
agent = PPOAgent(9, 1)

# InvertedDoublePendulum environment: maximize balance time up to 1000 steps
env = gym.make('InvertedDoublePendulum-v5')

os.makedirs('./data/models', exist_ok=True)
os.makedirs('./data/plots', exist_ok=True)

NUM_BUFFER_FILLS = 500
MAX_RECENT_EPISODES = 50

obs, info = env.reset()
episode_reward = 0.0
episode_rewards = []
avg_rewards = []
episode_count = 0
recent_rewards = deque(maxlen=MAX_RECENT_EPISODES)

for fill_idx in range(NUM_BUFFER_FILLS):
    # Collect trajectories until buffer is full (4096 steps)
    per_fill_rewards = []
    for _ in range(agent.buffer.buffer.maxlen):
        # Get action and log-prob from current policy
        z, action, log_prob, value = agent.step(obs, training=True)

        # Execute action in environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        # Bootstrap: zero value at true terminal, otherwise estimate from next state
        if terminated:
            next_value = 0.0
        else:
            next_value = agent.get_value(next_obs)

        # Store experience for GAE computation
        agent.store_experience(obs, z, value, next_value, reward, log_prob, terminated)
        
        # Handle episode termination (natural end or time limit)
        if terminated or truncated:
            episode_rewards.append(episode_reward)
            per_fill_rewards.append(episode_reward)
            episode_count += 1
            recent_rewards.append(episode_reward)
            running_avg = sum(recent_rewards) / len(recent_rewards)
            # Print episode performance every 10 episodes
            if episode_count % 10 == 0:
                print(
                    f'[Buffer {fill_idx + 1}/{NUM_BUFFER_FILLS}] '
                    f'Episode {episode_count} reward={episode_reward:.2f} '
                    f'running_avg_last{len(recent_rewards)}={running_avg:.2f}'
                )
            episode_reward = 0.0
            obs, info = env.reset()
        else:
            obs = next_obs

    # Perform PPO updates on collected rollout
    agent.optimize_model()

    # Clear buffer for next rollout collection
    agent.clear_buffer()

    # Record average episode reward for this buffer (not cumulative)
    if per_fill_rewards:
        avg_rewards.append(sum(per_fill_rewards) / len(per_fill_rewards))
    else:
        avg_rewards.append(0.0)

    # Print completion message for this buffer fill
    print(
        f'Completed buffer fill {fill_idx + 1}/{NUM_BUFFER_FILLS}: '
        f'total_episodes={episode_count}, overall_avg_reward={avg_rewards[-1]:.2f}'
    )

    # Only save model and plots every 50 buffer fills or at the last buffer fill
    if (fill_idx + 1) % 50 == 0 or (fill_idx + 1) == NUM_BUFFER_FILLS:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        checkpoint_path = f'./data/models/ppo_agent_{timestamp}.pth'
        agent.save_model(checkpoint_path)

        with open('./data/latest_checkpoint.txt', 'w') as f:
            f.write(checkpoint_path)

        fig = plt.figure(figsize=(8, 4))
        plt.plot(avg_rewards, marker='o', linewidth=2)
        plt.xlabel('Buffer fill')
        plt.ylabel('Average episode return')
        plt.title('Training performance')
        plt.grid(True)
        plt.tight_layout()
        fig.savefig(f'./data/plots/training_rewards_{timestamp}.png')
        fig.savefig(f'./data/plots/latest_training_rewards.png')
        plt.close(fig)