"""
Policy Network for Module Selection.

This module implements the policy network used by the PPO algorithm
to select modules and routing probabilities for the evolving architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    """
    Policy network for reinforcement learning-based architecture search.
    
    Takes the current state (step, modules, paths, etc.) and outputs:
    - Module selection logits
    - Routing probability parameters
    - Value estimate for the state
    """
    
    def __init__(self, hidden_dim: int = 128):
        """
        Initialize the policy network.
        
        Args:
            hidden_dim: Dimension of hidden layers
        """
        super(PolicyNetwork, self).__init__()
        
        # Shared feature extraction layers
        self.fc1 = nn.Linear(14, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # Module selection heads (2 action groups with 4 choices each)
        self.module_head1 = nn.Linear(hidden_dim, 8)  # First action group
        self.module_head2 = nn.Linear(hidden_dim, 8)  # Second action group
        
        # Value function head
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.module_head3 = nn.Linear(hidden_dim, 1)
        
        # Learnable standard deviation for routing probabilities
        self.std = nn.Parameter(torch.zeros((1, 1)))
    
    def forward(self, state: torch.Tensor):
        """
        Forward pass to compute action distributions and value.
        
        Args:
            state: State tensor of shape (batch_size, state_dim)
        
        Returns:
            Tuple of:
            - action_logits1: Logits for first action group (batch_size, 4, 2)
            - action_logits2: Logits for second action group (batch_size, 4, 2)
            - std: Standard deviation for routing probabilities (batch_size, 4)
            - value: State value estimate (batch_size, 1)
        """
        # Shared feature extraction
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Action selection (module choice)
        action_features = F.relu(self.fc3(x))
        action_logits1 = self.module_head1(action_features)
        action_logits2 = self.module_head2(action_features)
        
        # Reshape to (batch_size, 4, 2) for 4 actions with 2 choices each
        action_logits1 = action_logits1.reshape(action_logits1.size(0), 4, 2)
        action_logits2 = action_logits2.reshape(action_logits2.size(0), 4, 2)
        
        # Value function
        value_features = F.relu(self.fc4(x))
        value = self.module_head3(value_features)
        
        # Standard deviation for continuous parameters (routing probabilities)
        # Use softplus to ensure positivity, scaled by 0.35
        std = 0.35 * F.softplus(self.std).expand(action_logits1.size(0), 4)
        
        return action_logits1, action_logits2, std, value
