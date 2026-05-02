import os
import sys
import numpy as np
from stable_baselines3 import PPO

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)
import snake_engine

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]
model_path = os.path.join("models", "ppo_snake2.zip")

MAX_WIDTH = 44
MAX_HEIGHT = 24

if os.path.exists(model_path):
    model = PPO.load(model_path)
    print(f"Modele charge avec succes depuis : {model_path}")
else:
    model = None
    print("Fichier .zip introuvable. Mouvements aleatoires.")

static_walls = None

def get_obs(state, is_player_1):
    global static_walls
    real_width = state.width
    real_height = state.height

    obs = np.zeros((4, MAX_HEIGHT, MAX_WIDTH), dtype=np.float32)
    
    obs[0] = 1.0

    if static_walls is None or static_walls.shape != (real_height, real_width):
        static_walls = np.zeros((real_height, real_width), dtype=np.float32)
        for y in range(real_height):
            for x in range(real_width):
                if state.grid.get(x, y).getType() == snake_engine.TileType.TYPE_WALL:
                    static_walls[y][x] = 1.0

    obs[0, :real_height, :real_width] = static_walls

    try:
        for app in state.grid.apples:
            if hasattr(app, 'y') and hasattr(app, 'x'):
                if 0 <= app.y < real_height and 0 <= app.x < real_width:
                    obs[1][app.y][app.x] = 1.0
                elif 0 <= app.x < real_height and 0 <= app.y < real_width:
                    obs[1][app.x][app.y] = 1.0
    except Exception:
        for y in range(real_height):
            for x in range(real_width):
                coord = snake_engine.Coord(x, y)
                if coord in state.grid.apples:
                    obs[1][y][x] = 1.0

    my_bots = state.bots1 if is_player_1 else state.bots2
    opp_bots = state.bots2 if is_player_1 else state.bots1

    for bot_id, bot in my_bots.items():
        for part in bot.body:
            if 0 <= part.y < real_height and 0 <= part.x < real_width:
                obs[2][part.y][part.x] = 1.0
            elif 0 <= part.x < real_height and 0 <= part.y < real_width:
                obs[2][part.x][part.y] = 1.0

    for bot_id, bot in opp_bots.items():
        for part in bot.body:
            if 0 <= part.y < real_height and 0 <= part.x < real_width:
                obs[3][part.y][part.x] = 1.0
            elif 0 <= part.x < real_height and 0 <= part.y < real_width:
                obs[3][part.x][part.y] = 1.0

    return obs

def run(state, is_player_1=True):
    my_bots = state.bots1 if is_player_1 else state.bots2
    
    if len(my_bots) == 0:
        return {}

    if model is None:
        return {bot_id: np.random.choice(DIRS) for bot_id in my_bots.keys()}

    obs = get_obs(state, is_player_1)
    
    action, _ = model.predict(obs, deterministic=True)
    action_str = DIRS[int(action)]
    
    return {bot_id: action_str for bot_id in my_bots.keys()}