import sys
import os
import pygame
import alphazero as agent

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)

import snake_engine

BROWN = (139, 69, 19)
BLUE = (0, 0, 255)
LIGHT_BLUE = (100, 149, 237)
RED = (255, 0, 0)
LIGHT_RED = (255, 100, 100)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def main(display=True):
    state = snake_engine.GameState()
    
    CELL_SIZE = 20
    WIDTH = state.width * CELL_SIZE
    HEIGHT = state.height * CELL_SIZE
    UI_HEIGHT = 40
    
    if display:
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT + UI_HEIGHT))
        pygame.display.set_caption("Snake AlphaZero - Auto Match")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont(None, 30)
    
    running = True
    paused = False
    history = [state.copy()]
    current_idx = 0
    
    while running:
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_LEFT and paused:
                        if current_idx > 0:
                            current_idx -= 1
                            state = history[current_idx].copy()
                    elif event.key == pygame.K_RIGHT and paused:
                        if current_idx < len(history) - 1:
                            current_idx += 1
                            state = history[current_idx].copy()

        if not paused and len(state.bots1) > 0 and len(state.bots2) > 0:
            if current_idx < len(history) - 1:
                history = history[:current_idx + 1]

            my_actions = agent.run(state, is_player_1=True)
            opp_actions = agent.run(state, is_player_1=False)
            
            state.step(my_actions, opp_actions)
            history.append(state.copy())
            current_idx += 1
        if current_idx==200 : running=False
            
        if display: 
            screen.fill(BLACK)
            
            for y in range(state.height):
                for x in range(state.width):
                    tile = state.grid.get(x, y)
                    if tile.getType() == snake_engine.TileType.TYPE_WALL:
                        pygame.draw.rect(screen, BROWN, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                    
                    coord = snake_engine.Coord(x, y)
                    if coord in state.grid.apples:
                        pygame.draw.rect(screen, YELLOW, (x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            score1 = sum(len(bot.body) for bot in state.bots1.values())
            score2 = sum(len(bot.body) for bot in state.bots2.values())
            
            for bot_id, bot in state.bots1.items():
                for i, part in enumerate(bot.body):
                    color = LIGHT_BLUE if i == 0 else BLUE
                    pygame.draw.rect(screen, color, (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            for bot_id, bot in state.bots2.items():
                for i, part in enumerate(bot.body):
                    color = LIGHT_RED if i == 0 else RED
                    pygame.draw.rect(screen, color, (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            
            pygame.draw.rect(screen, (30, 30, 30), (0, HEIGHT, WIDTH, UI_HEIGHT))
            score_text = font.render(f"J1: {score1} | J2: {score2} | Frame: {current_idx}", True, WHITE)
            screen.blit(score_text, (10, HEIGHT + 10))
            
            if paused:
                pause_text = font.render("PAUSE", True, YELLOW)
                screen.blit(pause_text, (WIDTH - 100, HEIGHT + 10))
            elif len(state.bots1) == 0 or len(state.bots2) == 0:
                end_text = font.render("FIN DU JEU", True, RED)
                screen.blit(end_text, (WIDTH - 120, HEIGHT + 10))
            
            pygame.display.flip()
            clock.tick(10)
    
    print(f"Partie terminée ! Survivants Joueur 1: {len(state.bots1)}, Survivants Joueur 2: {len(state.bots2)}")
    
    if display:
        pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main(True)