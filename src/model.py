import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )

        self.mu_head = nn.Linear(256, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, state):
        shared_out = self.shared(state)
        mu = self.mu_head(shared_out)

        log_std = self.log_std.clamp(-20, 2)
        std = torch.exp(log_std)
        
        dist = Normal(mu, std)
        value = self.critic(shared_out)
        return dist, value