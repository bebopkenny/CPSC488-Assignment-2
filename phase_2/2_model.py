import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        # if numpy, convert; if sparse, ensure dense float32
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        else:
            x = x.float()
        return self.net(x)
