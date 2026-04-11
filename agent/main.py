import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)
import snake_engine

from coach import Coach
from env import SnakeEnv

def main():
    print("Initiating AlphaZero Training for Snake C++ Engine...")
    
    # Initialize PyBind C++ simulator
    c_state = snake_engine.GameState()
    env_wrapper = SnakeEnv(c_state)
    
    coach = Coach(env_wrapper, action_size=4) # UP, DOWN, LEFT, RIGHT per step
    coach.learn()

if __name__ == "__main__":
    main()
