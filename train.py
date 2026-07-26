import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from env import PongEnv, Action
from q_network import QNetwork

LR = 0.001
MEMORY_SIZE = 2000
BATCH_SIZE = 32
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
GAMMA = 0.95
EPISODES = 1000
TARGET_UPDATE_FREQ = 10 

def train_dqn() -> None:
    """Train a Deep Q-Network training loop for the Pong agent.
    
    Implement an epsilon-greedy strategy for exploration, experience replay
    batching for memory and a target network for stable learning. 
    """
    env = PongEnv()

    # Initialize target networks
    q_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    print("Starting DQN training...")

    for episode in range(1, EPISODES + 1):
        state = torch.tensor(env.reset(), dtype=torch.float32)
        game_over = False

        while not game_over:
            # Epsilon-greedy action selection
            if random.random() <= epsilon:
                action_val = random.choice(list(Action)).value
            else:
                with torch.no_grad():
                    q_values = q_net(state)
                    action_val = torch.argmax(q_values).item()

            # Take a step in the environment
            action = Action(action_val)
            next_state_list, reward, game_over = env.step(action)
            next_state = torch.tensor(next_state_list, dtype=torch.float32)

            # Save experience
            memory.append((state, action_val, reward, next_state, game_over))
            state = next_state

            if episode % 50 == 0:
                env.render()

            # Experience replay
            if len(memory) > BATCH_SIZE:
                minibatch = random.sample(memory, BATCH_SIZE)

                # Stack memories into tensors
                m_states = torch.stack([m[0] for m in minibatch])
                m_actions = torch.tensor([m[1] for m in minibatch])
                m_rewards = torch.tensor([m[2] for m in minibatch], dtype=torch.float32)
                m_next_states = torch.stack([m[3] for m in minibatch])
                m_dones = torch.tensor([m[4] for m in minibatch], dtype=torch.bool)

                # Predict Q-values for current states
                current_q_values = q_net(m_states)

                # Predict next Q-values using the target network
                with torch.no_grad():
                    next_q_values = target_net(m_next_states)

                # Calculate Bellman targets
                targets = current_q_values.clone()
                for i in range(BATCH_SIZE):
                    if m_dones[i]:
                        targets[i, m_actions[i]] = m_rewards[i]
                    else:
                        targets[i, m_actions[i]] = m_rewards[i] + GAMMA * torch.max(next_q_values[i]).item()

                # Single backpropagation pass
                loss = loss_fn(current_q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY

        # Target network update
        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(q_net.state_dict())
            
        if episode % 10 == 0:
            print(f"Episode {episode}/{EPISODES} | Score: {env.score} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train_dqn()