# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 00:28:52 2026

@author: Romain
"""

import random

def run(state, is_player_1=True):
    actions = {}
    bots = state.bots1 if is_player_1 else state.bots2
    
    possible_moves = ["UP", "DOWN", "LEFT", "RIGHT"]
    
    for bot_id, bot in bots.items():
        actions[bot_id] = random.choice(possible_moves)
        
    return actions