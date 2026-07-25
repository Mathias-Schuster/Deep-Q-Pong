import random
import time
from typing import List, Tuple
from enum import IntEnum

class Action(IntEnum):
    LEFT = 0
    STAY = 1
    RIGHT = 2

class PongEnv:
    """A Reinforcement Learning environment for a Pong-style game.
    
    The agent controls a paddle to keep a bouncing ball from hitting the ground.
    
    Args:
        width (int): The width of the game grid. Default is 11.
        height (int): The height of the game grid. Default is 7.
    """

    def __init__(self, width: int = 11, height: int = 7) -> None:
        self.width = width
        self.height = height
        self.reset()

    def reset(self) -> List[int]:
        """Reset the environment to its initial state and return it."""
        
        self.paddle_x = self.width // 2 
        self.ball_x = self.width // 2
        self.ball_y = 1
        
        self.ball_dx = random.choice([-1, 0, 1]) 
        self.ball_dy = 1 
        
        self.game_over = False
        self.score = 0
        return self.get_state()
    
    def get_state(self) -> List[int]:
        """Return the current state of the environment as a list of positions and velocity."""
        return [self.paddle_x, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy]

    def step(self, action: Action) -> Tuple[List[int], int, bool]:
        """Advance the environment by one timestep based on the agent's action.
        
        Update paddle position, ball position, handle collisions (wall/ceiling/paddle),
        and determine the reward. The game ends if the ball hits the ground.

        Return the new state, the reward, and whether the game is over.
        """
        if action == Action.LEFT and self.paddle_x > 1:
            self.paddle_x -= 1
        elif action == Action.RIGHT and self.paddle_x < self.width - 2:
            self.paddle_x += 1

        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Wall Collisions
        if self.ball_x <= 0:
            self.ball_x = 0
            self.ball_dx = 1
        elif self.ball_x >= self.width - 1:
            self.ball_x = self.width - 1
            self.ball_dx = -1
            
        if self.ball_y <= 0:
            self.ball_y = 0
            self.ball_dy = 1

        # Paddle Collision
        reward = 0
        if self.ball_y == self.height - 1:
            if self.paddle_x - 1 <= self.ball_x <= self.paddle_x + 1:
                reward = 10
                self.score += 1
                self.ball_dy = -1
                
                if self.ball_x < self.paddle_x: 
                    self.ball_dx = -1
                elif self.ball_x > self.paddle_x:
                    self.ball_dx = 1
                else:
                    self.ball_dx = 0
            else:
                reward = -10
                self.game_over = True

        return self.get_state(), reward, self.game_over

    def render(self) -> None:
        """Render the current state of the environment to the console."""
        print("\n" * 15)
        for y in range(self.height):
            row = ""
            for x in range(self.width):
                if y == self.ball_y and x == self.ball_x:
                    row += " o "
                elif y == self.height - 1 and (self.paddle_x - 1 <= x <= self.paddle_x + 1):
                    if x < self.paddle_x: row += " < "
                    elif x == self.paddle_x: row += " = "
                    else: row += " > "
                else:
                    row += " . "
            print(row)
        
        print(f"\nScore: {self.score}")
        time.sleep(0.2)


# Test game manually
if __name__ == "__main__":
    env = PongEnv()
    env.reset()
    game_over = False
    
    while not game_over:
        env.render()
        random_action = random.choice(list(Action)) 
        state, reward, game_over = env.step(random_action)
        
    env.render()
    print("Game Over!")