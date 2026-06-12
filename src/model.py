import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        def mlp(input_dim, output_dim):
            return nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, output_dim)
            )
        
        self.actor = mlp(state_dim, action_dim)
        self.critic = mlp(state_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mu = self.actor(state)
        value = self.critic(state)
        log_std = self.log_std.clamp(-20, 2)
        std = torch.exp(log_std)
        
        return Normal(mu, std), value