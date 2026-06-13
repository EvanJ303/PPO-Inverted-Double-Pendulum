import torch
from collections import deque, namedtuple
from model import ActorCritic

# Experience tuple stores trajectories for PPO training
experience = namedtuple('Experience', ['state', 'z', 'value', 'next_value', 'reward', 'log_prob', 'done'])

class rollout_buffer:
    """Fixed-size buffer for collecting rollout experiences."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.buffer = rollout_buffer(capacity=4096)

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

        self.gamma = 0.99
        self.epsilon = 0.2
        self.gae_lambda = 0.95
        self.lr = 3e-4
        self.batch_size = 128
        self.num_epochs = 8
        self.kl_threshold = 0.03

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def step(self, state, training):
        """Sample action and compute log-prob with tanh correction for continuous control."""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            dist, value = self.model(state)

            if training:
                z = dist.sample()
                action = torch.tanh(z)
                log_prob_z = dist.log_prob(z).sum(dim=-1)
                # Jacobian correction for tanh transformation
                log_prob = log_prob_z - torch.log(1 - action.pow(2) + 1e-8).sum(dim=-1)

                return (
                    z.squeeze(0).detach().cpu().numpy(),
                    action.squeeze(0).detach().cpu().numpy(),
                    log_prob.squeeze().detach().cpu().item(),
                    value.squeeze().detach().cpu().item(),
                )
            else:
                z = dist.mean
                action = torch.tanh(z)

                return action.squeeze(0).detach().cpu().numpy()
        
    def get_value(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            _, value = self.model(state)
            return value.squeeze().detach().cpu().item()

    def store_experience(self, state, z, value, next_value, reward, log_prob, done):
        exp = experience(state, z, value, next_value, reward, log_prob, done)
        self.buffer.add(exp)

    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation (GAE) for temporal-difference learning."""
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            # TD residual: r_t + γV(s_{t+1}) - V(s_t), zero bootstrap at episode end
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
            # Accumulate GAE with exponential decay, reset at episode boundaries
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        return advantages

    def optimize_model(self):
        rollout = list(self.buffer.buffer)
        states = torch.FloatTensor([exp.state for exp in rollout]).to(self.device)
        z_values = torch.FloatTensor([exp.z for exp in rollout]).to(self.device)
        values = torch.FloatTensor([exp.value for exp in rollout]).to(self.device)
        next_values = torch.FloatTensor([exp.next_value for exp in rollout]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in rollout]).to(self.device)
        log_probs = torch.FloatTensor([exp.log_prob for exp in rollout]).to(self.device)
        dones = torch.FloatTensor([exp.done for exp in rollout]).to(self.device)

        advantages = self.compute_gae(rewards, values, next_values, dones)
        advantages = torch.FloatTensor(advantages).to(self.device)

        # Compute returns (advantage + baseline value) for critic loss
        returns = advantages + values

        # Normalize advantages to reduce variance and improve gradient stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.num_epochs):
            kl_exceeded = False
            indices = torch.randperm(len(rollout))

            for i in range(0, len(rollout), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = states[batch_indices]
                batch_z = z_values[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                dist, value = self.model(batch_states)
                new_log_probs_z = dist.log_prob(batch_z).sum(dim=-1)
                # Apply same tanh correction as in step() for consistency
                new_log_probs = new_log_probs_z - torch.log(1 - torch.tanh(batch_z).pow(2) + 1e-8).sum(dim=-1)

                # Early stopping: if policy change is too large, skip remaining epochs
                approx_kl = (batch_log_probs - new_log_probs).mean()
                if approx_kl > self.kl_threshold:
                    kl_exceeded = True
                    break

                # PPO clipped surrogate objective
                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                # Take minimum to clip policy update
                actor_loss = -torch.min(surr1, surr2).mean()

                # MSE loss for value function approximation
                critic_loss = (batch_returns - value.squeeze(1)).pow(2).mean()

                # Entropy encourages exploration
                entropy_loss = dist.entropy().sum(dim=-1).mean()

                # Total loss: policy + value + entropy regularization
                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
            
            if kl_exceeded:
                break

    def clear_buffer(self):
        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))