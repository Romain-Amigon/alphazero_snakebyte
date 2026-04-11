import sys
import os
import torch
import pygame
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)

import snake_engine
from train import AlphaZeroNet
from mcts import run_mcts
import random_bot

def main():
    # En SSH, on force Pygame a utiliser un affichage "dummy" (fantome)
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = AlphaZeroNet(channels=4).to(device)
    
    model_path = os.path.join(current_dir, "models", "alphazero_snake_ep20.pth")
    if os.path.exists(model_path):
        net.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Modèle chargé avec succès depuis: {model_path}")
    else:
        print("ATTENTION: Modèle non trouvé, poids aléatoires.")
        
    net.eval()
    
    state = snake_engine.GameState()
    CELL_SIZE = 20
    WIDTH = state.width * CELL_SIZE
    HEIGHT = state.height * CELL_SIZE

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))

    BROWN = (139, 69, 19)
    BLUE = (0, 0, 255)
    LIGHT_BLUE = (100, 149, 237)
    RED = (255, 0, 0)
    LIGHT_RED = (255, 100, 100)
    YELLOW = (255, 255, 0)
    BLACK = (0, 0, 0)

    # Configuration du pipeline FFmpeg
    output_fps = 2  # ralenti pour voir les tours
    output_file = os.path.join(current_dir, "match_alphaZero_vs_Random.mp4")
    
    process = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-s', f"{WIDTH}x{HEIGHT}",
        '-pix_fmt', 'rgb24', '-r', str(output_fps),
        '-i', '-', '-vcodec', 'libx264', '-crf', '18',
        '-preset', 'fast', '-pix_fmt', 'yuv420p',
        output_file
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    tour = 0
    running = True
    print("Début de l'enregistrement de la partie (sans interface)...")

    my_tree = None

    while running:
        if len(state.bots1) > 0 and len(state.bots2) > 0 and len(state.grid.apples) > 0 and tour < 200:
            my_actions, _, my_tree = run_mcts(state, net, mcts_sims=50, is_p1=True, tree=my_tree)
            opp_actions = random_bot.run(state, is_player_1=False)
            
            state.step(my_actions, opp_actions)
            tour += 1
            if tour % 10 == 0:
                print(f"  Tour {tour}/200...")
        else:
            if len(state.bots1) > 0 and len(state.bots2) == 0:
                print(f"Fin: AlphaZero Bleu a gagné ! (Tours: {tour})")
            elif len(state.bots2) > 0 and len(state.bots1) == 0:
                print(f"Fin: Random Rouge a gagné ! (Tours: {tour})")
            else:
                print(f"Fin: Match Nul ou limite de tours. (Tours: {tour})")
            running = False

        screen.fill(BLACK)

        # Affichage
        for y in range(state.height):
            for x in range(state.width):
                tile = state.grid.get(x, y)
                if tile.getType() == snake_engine.TileType.TYPE_WALL:
                    pygame.draw.rect(screen, BROWN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                
                coord = snake_engine.Coord(x, y)
                if coord in state.grid.apples:
                    pygame.draw.rect(screen, YELLOW, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for bot_id, bot in state.bots1.items():
            for i, part in enumerate(bot.body):
                pygame.draw.rect(screen, LIGHT_BLUE if i == 0 else BLUE, (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        
        for bot_id, bot in state.bots2.items():
            for i, part in enumerate(bot.body):
                pygame.draw.rect(screen, LIGHT_RED if i == 0 else RED, (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        # Envoi de la frame actuelle à FFmpeg
        frame = pygame.surfarray.pixels3d(screen)
        # pygame.surfarray returns (width, height, 3) Mais rawvideo attend (height, width, 3) 
        # pygame array est en fait RGB mais formaté (X,Y). FFmpeg attend des bytes continus sur les lignes(X), hauteur (Y)
        import numpy as np
        frame = np.transpose(frame, (1, 0, 2))
        process.stdin.write(frame.tobytes())
        
        # Le array verrouille la surface, on devrait l'effacer explicitement en pure theorie, numpy dereference
        del frame

    process.stdin.close()
    process.wait()
    pygame.quit()
    print(f"\n=> Video MP4 générée avec succès : {output_file}")
    
if __name__ == '__main__':
    main()
