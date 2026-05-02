import sys
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)
import snake_engine

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

MAX_WIDTH = 44
MAX_HEIGHT = 24

class SnakeEnv(gym.Env):
    def __init__(self):
        super(SnakeEnv, self).__init__()
        
        self.state = snake_engine.GameState()
        
        self.action_space = spaces.Discrete(4)
        
        # L'espace d'observation est maintenant FIXÉ au maximum
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(4, MAX_HEIGHT, MAX_WIDTH), 
            dtype=np.float32
        )
        
        self.current_tour = 0
        self.max_tours = 200
        self.last_score1 = 0
        self.static_walls = None

    def _get_obs(self):
        # On crée un tenseur vide de la taille MAXIMALE
        obs = np.zeros((4, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
        
        real_width = self.state.width
        real_height = self.state.height
        
        # 1. On remplit tout l'espace hors-limites avec des murs (1.0)
        obs[0] = 1.0 
        
        # 2. On creuse notre vraie zone de jeu
        if self.static_walls is None or self.static_walls.shape != (real_height, real_width):
            self.static_walls = np.zeros((real_height, real_width), dtype=np.float32)
            for y in range(real_height):
                for x in range(real_width):
                    if self.state.grid.get(x, y).getType() == snake_engine.TileType.TYPE_WALL:
                        self.static_walls[y][x] = 1.0
                        
        # On insère nos vrais murs dans le coin en haut à gauche
        obs[0, :real_height, :real_width] = self.static_walls
        
        # 3. On place les pommes (avec le bouclier anti-crash)
        try:
            for app in self.state.grid.apples:
                if hasattr(app, 'y') and hasattr(app, 'x'):
                    if 0 <= app.y < real_height and 0 <= app.x < real_width:
                        obs[1][app.y][app.x] = 1.0
                    elif 0 <= app.x < real_height and 0 <= app.y < real_width:
                        obs[1][app.x][app.y] = 1.0
        except Exception:
            for y in range(real_height):
                for x in range(real_width):
                    coord = snake_engine.Coord(x, y)
                    if coord in self.state.grid.apples:
                        obs[1][y][x] = 1.0
            
        # 4. On place les bots
        for bot_id, bot in self.state.bots1.items():
            for part in bot.body:
                if 0 <= part.y < real_height and 0 <= part.x < real_width:
                    obs[2][part.y][part.x] = 1.0
                elif 0 <= part.x < real_height and 0 <= part.y < real_width:
                    obs[2][part.x][part.y] = 1.0
                    
        for bot_id, bot in self.state.bots2.items():
            for part in bot.body:
                if 0 <= part.y < real_height and 0 <= part.x < real_width:
                    obs[3][part.y][part.x] = 1.0
                elif 0 <= part.x < real_height and 0 <= part.y < real_width:
                    obs[3][part.x][part.y] = 1.0
                    
        return obs

    # ... Le reste de tes fonctions reset() et step() ne change pas ...

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = snake_engine.GameState()
        self.current_tour = 0
        self.last_score1 = sum(len(bot.body) for bot in self.state.bots1.values())
        return self._get_obs(), {}

    def step(self, action):
        my_action_str = DIRS[int(action)]
        my_actions = {bot_id: my_action_str for bot_id in self.state.bots1.keys()}
        
        opp_actions = {}
        for bot_id in self.state.bots2.keys():
            opp_actions[bot_id] = np.random.choice(DIRS)
            
        self.state.step(my_actions, opp_actions)
        self.current_tour += 1
        
        score1 = sum(len(bot.body) for bot in self.state.bots1.values())
        
        terminated = False
        truncated = False
        reward = 0.0
        
        has_apples = False
        try:
            has_apples = len(self.state.grid.apples) > 0
        except TypeError:
            has_apples_count = 0
            for y in range(self.height):
                for x in range(self.width):
                    if snake_engine.Coord(x, y) in self.state.grid.apples:
                        has_apples_count += 1
            has_apples = has_apples_count > 0

        if len(self.state.bots1) == 0:
            terminated = True
            reward = -1.0
        elif len(self.state.bots2) == 0:
            terminated = True
            reward = 1.0
        elif not has_apples:
            terminated = True
            score2 = sum(len(bot.body) for bot in self.state.bots2.values())
            if score1 > score2:
                reward = 1.0
            elif score1 < score2:
                reward = -1.0
            else:
                reward = 0.0
        else:
            if score1 > self.last_score1:
                reward = 0.1 
            self.last_score1 = score1
            
            if self.current_tour >= self.max_tours:
                truncated = True
                
        obs = self._get_obs()
        return obs, reward, terminated, truncated, {}

def train_ppo():
    print("Demarrage de l'entrainement PPO...")
    
    env = SnakeEnv()
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        normalize_images=False,
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tensorboard_logs/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    model.learn(total_timesteps=1000000)
    
    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "ppo_snake2")
    model.save(save_path)
    print(f"Modele sauvegarde dans : {save_path}.zip")

if __name__ == "__main__":
    train_ppo()