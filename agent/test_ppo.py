import os
import sys
import glob
import numpy as np
import subprocess
import pygame

try:
    import snake_engine
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "..", "build", "Release"))
    import snake_engine

from stable_baselines3 import PPO
from train_ppo import CustomMultiExtractor
from ppo_env import SnakePPOEnv

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]
INT_TO_DIR = {i: d for i, d in enumerate(DIRS)}

CELL_SIZE = 20
BROWN      = (139, 69, 19)
BLUE       = (0, 0, 255)
LIGHT_BLUE = (100, 149, 237)
RED        = (255, 0, 0)
LIGHT_RED  = (255, 100, 100)
YELLOW     = (255, 255, 0)
BLACK      = (0, 0, 0)


def find_latest_checkpoint(folder="models_v2"):
    zips = glob.glob(os.path.join(folder, "*.zip"))
    if not zips:
        return None
    return max(zips, key=os.path.getmtime)


def render_frame(state, screen, ffmpeg_proc):
    screen.fill(BLACK)

    for y in range(state.height):
        for x in range(state.width):
            if state.grid.get(x, y).getType() == 1:
                pygame.draw.rect(screen, BROWN,
                                 (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for apple in state.grid.apples:
        pygame.draw.rect(screen, YELLOW,
                         (apple.x * CELL_SIZE, apple.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for bot_id, bot in state.bots1.items():
        for i, part in enumerate(bot.body):
            color = LIGHT_BLUE if i == 0 else BLUE
            pygame.draw.rect(screen, color,
                             (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for bot_id, bot in state.bots2.items():
        for i, part in enumerate(bot.body):
            color = LIGHT_RED if i == 0 else RED
            pygame.draw.rect(screen, color,
                             (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    frame = np.transpose(pygame.surfarray.pixels3d(screen), (1, 0, 2))
    ffmpeg_proc.stdin.write(frame.tobytes())
    del frame


def run_games(model_path=None, num_games=5, fps=4, out_dir=None):
    if model_path is None:
        model_path = find_latest_checkpoint()
    if model_path is None:
        print("Aucun checkpoint trouvé dans models_v2/")
        return
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Modèle : {model_path}")

    policy_kwargs = dict(
        features_extractor_class=CustomMultiExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )
    model = PPO.load(model_path, custom_objects={"policy_kwargs": policy_kwargs})
    print("Modèle chargé.\n")

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    env = SnakePPOEnv(max_steps=500)

    for game_idx in range(num_games):
        env.reset()
        state = env.state

        W = state.width  * CELL_SIZE
        H = state.height * CELL_SIZE
        screen = pygame.display.set_mode((W, H))

        out_file = os.path.join(out_dir, f"ppo_match_{game_idx + 1}.mp4")
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
             "-s", f"{W}x{H}", "-pix_fmt", "rgb24", "-r", str(fps),
             "-i", "-", "-vcodec", "libx264", "-crf", "18",
             "-preset", "fast", "-pix_fmt", "yuv420p", out_file],
            stdin=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

        render_frame(state, screen, ffmpeg)
        steps = 0

        while steps < 500 and len(state.bots1) > 0 and len(state.bots2) > 0:
            # Chaque bot de l'équipe PPO joue indépendamment
            my_actions = {}
            for b_id in list(state.bots1.keys()):
                obs = env._get_obs(override_focus_id=b_id)
                action, _ = model.predict(obs, deterministic=True)
                my_actions[b_id] = INT_TO_DIR[int(action)]

            opp_actions = {b_id: env._get_smart_opp_action(b_id)
                           for b_id in state.bots2.keys()}

            state.step(my_actions, opp_actions)
            env.current_step += 1
            steps += 1

            render_frame(state, screen, ffmpeg)

        ffmpeg.stdin.close()
        ffmpeg.wait()

        score1 = sum(len(b.body) for b in state.bots1.values())
        score2 = sum(len(b.body) for b in state.bots2.values())

        if len(state.bots1) > 0 and len(state.bots2) == 0:
            result = f"VICTOIRE PPO (élimination, score {score1}-{score2})"
        elif len(state.bots2) > 0 and len(state.bots1) == 0:
            result = f"DÉFAITE PPO (élimination, score {score1}-{score2})"
        else:
            # Timeout : on départage au score (longueur cumulée des corps)
            if score1 > score2:
                result = f"VICTOIRE PPO aux points ({score1}-{score2}, {steps} tours)"
            elif score2 > score1:
                result = f"DÉFAITE PPO aux points ({score1}-{score2}, {steps} tours)"
            else:
                result = f"MATCH NUL ({score1}-{score2}, {steps} tours)"

        print(f"Partie {game_idx + 1} — {result} — vidéo : {out_file}")

    pygame.quit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Chemin vers le .zip")
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--fps",   type=int, default=4)
    args = parser.parse_args()

    run_games(model_path=args.model, num_games=args.games, fps=args.fps)
