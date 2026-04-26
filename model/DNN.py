import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Cube import TARGET_STATE_ONE_HOT


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class DNN(nn.Module):
    def __init__(self, input_dim, num_residual_blocks=4):
        super(DNN, self).__init__()

        # First two hidden layers
        self.fc1 = nn.Linear(input_dim, 5000)
        self.bn1 = nn.BatchNorm1d(5000)
        self.fc2 = nn.Linear(5000, 1000)
        self.bn2 = nn.BatchNorm1d(1000)

        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_residual_blocks):
            self.residual_blocks.append(ResidualBlock(1000, 1000))

        # Output layer
        self.output_layer = nn.Linear(1000, 1)

    def forward(self, x):
        # First two hidden layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))

        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Output layer
        x = self.output_layer(x)

        return x  # * self.K


# Example usage
if __name__ == "__main__":
    # Assume input dimension is 54*6=324 (Rubik's cube state encoding from Readme)
    input_dim = 324
    model = DNN(input_dim, num_residual_blocks=4)
    print(model)

    # Test forward pass
    test_input = torch.randn(10, input_dim)
    output = model(test_input)
    print(f"Input shape: {test_input.shape}")
    print(f"Output shape: {output.shape}")
