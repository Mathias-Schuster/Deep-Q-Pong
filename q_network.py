import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """A Multi-Layer Perceptron (MLP) for Deep Q-Learning.
    
    Maps state representations to Q-values for each possible action.
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 24, output_dim: int = 3) -> None:
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the network."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Sanity Test
if __name__ == "__main__":
    q_net = QNetwork()
    
    dummy_state = torch.tensor([5.0, 5.0, 1.0, 0.0, 1.0], dtype=torch.float32)
    
    with torch.no_grad():
        q_values = q_net(dummy_state)
        
    print(f"Q-values [Left, Stay, Right]: {q_values.numpy()}")
    
    best_action_idx = torch.argmax(q_values).item()
    action_names = ["Move Left", "Stay", "Move Right"]
    print(f"Untrained network selects action: {action_names[best_action_idx]}")