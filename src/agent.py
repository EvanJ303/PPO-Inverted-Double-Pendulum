import torch
from collections import deque, namedtuple
from model import ActorCritic

experience = namedtuple('Experience', ['state', 'z', 'value', 'next_value', 'reward', 'log_prob', 'done'])

class rollout_buffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.model = ActorCritic(state_dim, action_dim)
        self.buffer = rollout_buffer(capacity=2048)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)

        self.gamma = 0.99
        self.epsilon = 0.2
        self.gae_lambda = 0.95
        self.lr = 3e-4
        self.batch_size = 64
        self.num_epochs = 10

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def step(self, state, stochastic=True):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            dist, value = self.model(state)

            if stochastic:
                z = dist.sample()
                action = torch.tanh(z)
                log_prob_z = dist.log_prob(z)
                log_prob = log_prob_z - torch.log(1 - action.pow(2) + 1e-7).sum(dim=-1)

                return (
                    z.squeeze(0).detach().cpu().numpy(),
                    action.squeeze(0).detach().cpu().numpy(),
                    float(log_prob.squeeze().detach().cpu().numpy()),
                    float(value.squeeze().detach().cpu().numpy()),
                )
            else:
                z = dist.mean()
                action = torch.tanh(z)

                return action.squeeze(0).detach().cpu().numpy()
        
    def get_value(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            _, value = self.model(state)
            return float(value.squeeze().detach().cpu().numpy())

    def store_experience(self, state, z, value, next_value, reward, log_prob, done):
        exp = experience(state, z, value, next_value, reward, log_prob, done)
        self.buffer.add(exp)

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_values[i] * (1 - dones[i]) - values[i]
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

        returns = advantages + values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.num_epochs):
            indices = torch.randperm(len(rollout))
            for i in range(0, len(rollout), self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                batch_states = states[batch_indices]
                batch_z = z_values[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                dist, value = self.model(batch_states)
                new_log_probs_z = dist.log_prob(batch_z)
                new_log_probs = new_log_probs_z - torch.log(1 - torch.tanh(batch_z).pow(2) + 1e-7).sum(dim=-1)

                approx_kl = (batch_log_probs - new_log_probs).mean().item()
                if approx_kl > 0.015:
                    return

                ratio = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = (batch_returns - value.squeeze(1)).pow(2).mean()

                entropy_loss = dist.entropy().mean()

                loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    def clear_buffer(self):
        self.buffer.clear()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))