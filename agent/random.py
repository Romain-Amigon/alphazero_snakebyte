# -*- coding: utf-8 -*-
import random
import snake_engine

# UP est corrigé pour faire du Y négatif (monter)
DIRS = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, -1),
    "DOWN": (0, 1)
}

def run(state, is_player_1=True):
    actions = {}
    bots = state.bots1 if is_player_1 else state.bots2
    
    for bot_id, bot in bots.items():
        possible_moves = [] # Vidé pour chaque nouveau robot
        
        # On lit les coordonnées de la tête
        x = bot.body[0].x
        y = bot.body[0].y
        
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            a, b = DIRS[d]
            nx, ny = x + a, y + b

            if 0 <= nx < state.width and 0 <= ny < state.height:
                coord = snake_engine.Coord(nx, ny)
                
                # Ajout des parenthèses à getType() !
                if state.grid.get(nx, ny).getType() == snake_engine.TileType.TYPE_EMPTY or coord in state.grid.apples:
                    possible_moves.append(d)
        
        # Sécurité : si le robot est coincé, on force un coup pour éviter le crash de random.choice
        if possible_moves:
            actions[bot_id] = random.choice(possible_moves)
        else:
            actions[bot_id] = "UP"
            
    return actions# -*- coding: utf-8 -*-
import random
import snake_engine

# UP est corrigé pour faire du Y négatif (monter)
DIRS = {
    "LEFT": (-1, 0),
    "RIGHT": (1, 0),
    "UP": (0, -1),
    "DOWN": (0, 1)
}

def run(state, is_player_1=True):
    actions = {}
    bots = state.bots1 if is_player_1 else state.bots2
    
    for bot_id, bot in bots.items():
        possible_moves = [] # Vidé pour chaque nouveau robot
        
        # On lit les coordonnées de la tête
        x = bot.body[0].x
        y = bot.body[0].y
        
        for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
            a, b = DIRS[d]
            nx, ny = x + a, y + b

            if 0 <= nx < state.width and 0 <= ny < state.height:
                coord = snake_engine.Coord(nx, ny)
                
                # Ajout des parenthèses à getType() !
                if state.grid.get(nx, ny).getType() == snake_engine.TileType.TYPE_EMPTY or coord in state.grid.apples:
                    possible_moves.append(d)
        
        # Sécurité : si le robot est coincé, on force un coup pour éviter le crash de random.choice
        if possible_moves:
            actions[bot_id] = random.choice(possible_moves)
        else:
            actions[bot_id] = "UP"
            
    return actions