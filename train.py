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

def train_dqn() -> None:
    """Train a Deep Q-Network with basic experience replay."""
    env = PongEnv()
    q_net = QNetwork()
    optimizer = optim.Adam(q_net.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    memory = deque(maxlen=MEMORY_SIZE)
    epsilon = EPSILON_START

    print("Starting DQN training...")

    for episode in range(EPISODES):
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
                
                for m_state, m_action, m_reward, m_next_state, m_done in minibatch:
                    target = m_reward
                    if not m_done:
                        with torch.no_grad():
                            target = m_reward + GAMMA * torch.max(q_net(m_next_state)).item()
                            
                    current_q_values = q_net(m_state)
                    target_q_values = current_q_values.clone()
                    target_q_values[m_action] = target 
                    
                    loss = loss_fn(current_q_values, target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
        if epsilon > EPSILON_MIN:
            epsilon *= EPSILON_DECAY
            
        if episode % 10 == 0:
            print(f"Episode {episode}/{EPISODES} | Score: {env.score} | Epsilon: {epsilon:.3f}")

if __name__ == "__main__":
    train_dqn()