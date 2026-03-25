import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)

import pygame
import snake_engine

state = snake_engine.GameState()

CELL_SIZE = 20
WIDTH = state.width * CELL_SIZE
HEIGHT = state.height * CELL_SIZE

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake AlphaZero")
clock = pygame.time.Clock()

BROWN = (139, 69, 19)
BLUE = (0, 0, 255)
LIGHT_BLUE = (100, 149, 237)
RED = (255, 0, 0)
LIGHT_RED = (255, 100, 100)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    my_actions = {}
    opp_actions = {}
    state.step(my_actions, opp_actions)

    screen.fill(BLACK)

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
            color = LIGHT_BLUE if i == 0 else BLUE
            pygame.draw.rect(screen, color, (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    for bot_id, bot in state.bots2.items():
        for i, part in enumerate(bot.body):
            color = LIGHT_RED if i == 0 else RED
            pygame.draw.rect(screen, color, (part.x * CELL_SIZE, part.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    pygame.display.flip()
    clock.tick(5)

pygame.quit()
sys.exit()