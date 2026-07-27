import time
import torch

from env import PongEnv, Action
from q_network import QNetwork

def watch_trained_agent(model_path: str = "best_model.pth") -> None:
    """Load a trained model and watch it play one episode."""
    env = PongEnv()
    q_net = QNetwork()
    
    try:
        q_net.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Successfully loaded {model_path}!")
    except FileNotFoundError:
        print(f"Error: Could not find '{model_path}'. Make sure the file exists in the current directory or run train.py first!")
        return

    q_net.eval() 
    state = torch.tensor(env.reset(), dtype=torch.float32)
    game_over = False
    
    while not game_over:
        env.render()
        
        with torch.no_grad():
            q_values = q_net(state)
            action_val = torch.argmax(q_values).item()
            
        action = Action(action_val)
        next_state_list, _, game_over = env.step(action)
        state = torch.tensor(next_state_list, dtype=torch.float32)
        
        time.sleep(0.05)
    
    env.render()
    
    print(f"\nGame Over! Final Score: {env.score}")

if __name__ == "__main__":
    watch_trained_agent()