# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 00:28:50 2026

@author: Romain
"""

import sys
import os
import pygame
import agent

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)

import snake_engine
import time

BROWN = (139, 69, 19)
BLUE = (0, 0, 255)
LIGHT_BLUE = (100, 149, 237)
RED = (255, 0, 0)
LIGHT_RED = (255, 100, 100)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

def main(display=True):
    
    state = snake_engine.GameState()
    
    CELL_SIZE = 20
    WIDTH = state.width * CELL_SIZE
    HEIGHT = state.height * CELL_SIZE
    
    if display :
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake AlphaZero - Auto Match")
        clock = pygame.time.Clock()
    
    
    
    running = True
    while running and len(state.bots1) > 0 and len(state.bots2) > 0:
        if display:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
    
        my_actions = agent.run(state, is_player_1=True)
        opp_actions = agent.run(state, is_player_1=False)
    
        state.step(my_actions, opp_actions)
    
        if display : 
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
            clock.tick(10)
            
            time.sleep(1)
    
    print(f"Partie terminée ! Survivants Joueur 1: {len(state.bots1)}, Survivants Joueur 2: {len(state.bots2)}")
    
    pygame.quit()
    sys.exit()

main(True)