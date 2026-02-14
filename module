"""
Expert Module Definitions for Modular Continual Learning.

This module contains the definitions for:
- InputHead: Processes raw input into feature representations
- ExpertModule: Reusable expert components
- OutputHead: Final classification layer
"""

import torch
import torch.nn as nn


class InputHead(nn.Module):
    """
    Input processing head that converts raw input to feature representations.
    
    Uses convolutional layers with normalization and activation to extract
    initial features from input data.
    """
    
    def __init__(self, config: dict):
        """
        Initialize the input head.
        
        Args:
            config: Configuration dictionary containing:
                - input_dim: Number of input channels
                - hidden_dim: Dimension of hidden features
        """
        super().__init__()
        reduced_dim = config['hidden_dim']
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                config['input_dim'], 
                reduced_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.GroupNorm(8, reduced_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, H, W)
            
        Returns:
            Feature tensor of shape (batch_size, hidden_dim, H, W)
        """
        return self.layers(x)


class ExpertModule(nn.Module):
    """
    Expert module that can be reused across tasks.
    
    Each expert module is a residual-like block with convolution,
    normalization, and activation. Experts can be frozen or fine-tuned
    depending on the task requirements.
    """
    
    def __init__(self, hidden_dim: int, norm_layer=nn.GroupNorm):
        """
        Initialize the expert module.
        
        Args:
            hidden_dim: Dimension of hidden features
            norm_layer: Normalization layer class (default: GroupNorm)
        """
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(
                hidden_dim, 
                hidden_dim, 
                kernel_size=3, 
                stride=1, 
                padding=1
            ),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim, H, W)
            
        Returns:
            Output tensor of shape (batch_size, hidden_dim, H, W)
        """
        return self.layers(x)


class OutputHead(nn.Module):
    """
    Output classification head.
    
    Converts feature representations to class predictions using
    adaptive pooling followed by fully connected layers.
    """
    
    def __init__(
        self, 
        hidden_dim: int, 
        num_classes: int, 
        dropout_rate: float = 0.2
    ):
        """
        Initialize the output head.
        
        Args:
            hidden_dim: Dimension of input features
            num_classes: Number of output classes
            dropout_rate: Dropout probability for regularization
        """
        super().__init__()
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, hidden_dim, H, W)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # Adaptive pooling handles arbitrary input sizes
        x = self.adaptive_pool(x)
        # Flatten
        x = x.view(x.size(0), -1)
        # Classification
        return self.classifier(x)
