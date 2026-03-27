# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 01:50:59 2026

@author: Romain
"""

import os
import sys
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)
import snake_engine

from mcts import run_mcts
from train import AlphaZeroNet, device

net = AlphaZeroNet(channels=4).to(device)
model_path = os.path.join("models", "alphazero_snake.pth")

if os.path.exists(model_path):
    net.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modele charge avec succes depuis : {model_path}")
else:
    print("Fichier .pth introuvable. Mouvements non optimises.")

net.eval()

def run(state, is_player_1=True):
    actions, _ = run_mcts(state, net, 25, is_player_1)
    return actions