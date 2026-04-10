import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir,"..", "build", "Release")
sys.path.append(build_dir)
import snake_engine

from mcts import run_mcts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_tensor(state):
    tensor = np.zeros((4, state.height, state.width), dtype=np.float32)
    for y in range(state.height):
        for x in range(state.width):
            if state.grid.get(x, y).getType() == snake_engine.TileType.TYPE_WALL:
                tensor[0][y][x] = 1.0
            coord = snake_engine.Coord(x, y)
            if coord in state.grid.apples:
                tensor[1][y][x] = 1.0
    for bot_id, bot in state.bots1.items():
        for part in bot.body:
            if 0 <= part.x < state.width and 0 <= part.y < state.height:
                tensor[2][part.y][part.x] = 1.0
    for bot_id, bot in state.bots2.items():
        for part in bot.body:
            if 0 <= part.x < state.width and 0 <= part.y < state.height:
                tensor[3][part.y][part.x] = 1.0
    return torch.tensor(tensor).unsqueeze(0).to(device)

class AlphaZeroNet(nn.Module):
    def __init__(self, channels):
        super(AlphaZeroNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        fc_size = 64 * 8 * 8
        self.policy_head = nn.Sequential(
            nn.Linear(fc_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.value_head = nn.Sequential(
            nn.Linear(fc_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

def train_alphazero():
    print(f"Execution sur : {device}")
    
    net = AlphaZeroNet(channels=4).to(device)
    #checkpoint = torch.load("models/alphazero_snake.pth", map_location=device)
    #net.load_state_dict(checkpoint)
    #print("Poids chargés avec succès !")
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    criterion_value = nn.MSELoss()
    log_softmax = nn.LogSoftmax(dim=1)

    epochs = 100
    games_per_epoch = 100
    mcts_sims = 25
    batch_size = 64
    memory = deque(maxlen=100000)
    dirs = ["UP", "DOWN", "LEFT", "RIGHT"]

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        for game in range(games_per_epoch):
            print(f"\r  Partie {game+1}/{games_per_epoch} en cours...", end="")
            sys.stdout.flush()
            
            state = snake_engine.GameState()
            game_history = []
            tour = 0

            while len(state.bots1) > 0 and len(state.bots2) > 0 and tour < 200 and len(state.grid.apples)>0:
                tensor_state = state_to_tensor(state)
                
                my_actions, my_visits = run_mcts(state, net, mcts_sims, True)
                opp_actions, opp_visits = run_mcts(state, net, mcts_sims, False)

                pi = np.zeros(4, dtype=np.float32)
                total_visits = sum(my_visits.values())
                if total_visits > 0:
                    for act_str, count in my_visits.items():
                        act_dict = eval(act_str)
                        for bot_id, d in act_dict.items():
                            pi[dirs.index(d)] += count
                    if np.sum(pi) > 0:
                        pi /= np.sum(pi)

                game_history.append((tensor_state, pi))
                state.step(my_actions, opp_actions)
                tour += 1
   
            score1 = sum(len(bot.body) for bot in state.bots1.values())
            score2 = sum(len(bot.body) for bot in state.bots2.values())

            reward = 0.0
            if len(state.bots1) > 0 and len(state.bots2) == 0:
                reward = 1.0
            elif len(state.bots2) > 0 and len(state.bots1) == 0:
                reward = -1.0
            elif (score1+score2) !=0:
                reward = (score1-score2)/(score1+score2)
            else : reward =0


            for hist_state, pi in game_history:
                memory.append((hist_state, pi, reward))
        
        print()

        net.train()
        total_loss = 0
        
        memory_list = list(memory)
        random.shuffle(memory_list)
        
        batches = [memory_list[i:i + batch_size] for i in range(0, len(memory_list), batch_size)]

        for batch in batches:
            states, pis, rewards = zip(*batch)
            
            max_h = max(s.shape[2] for s in states)
            max_w = max(s.shape[3] for s in states)
            padded_states = []
            
            for s in states:
                pad_h = max_h - s.shape[2]
                pad_w = max_w - s.shape[3]
                padded_states.append(F.pad(s, (0, pad_w, 0, pad_h)))
                
            batch_state = torch.cat(padded_states, dim=0)
            batch_pi = torch.tensor(np.array(pis), dtype=torch.float32).to(device)
            batch_reward = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = net(batch_state)
            
            value_loss = criterion_value(pred_value, batch_reward)
            policy_loss = -torch.sum(batch_pi * log_softmax(pred_policy)) / len(batch)
            loss = value_loss + policy_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Loss: {total_loss/len(batches) if batches else 0:.4f}")

    os.makedirs("models", exist_ok=True)
    save_path = os.path.join("models", "alphazero_snake2.pth")
    torch.save(net.state_dict(), save_path)
    print(f"\nModele sauvegarde avec succes dans : {save_path}")

if __name__ == "__main__":
    train_alphazero()