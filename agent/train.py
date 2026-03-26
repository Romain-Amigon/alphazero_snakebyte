import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "build", "Release")
sys.path.append(build_dir)
import snake_engine

from mcts import run_mcts

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
            tensor[2][part.y][part.x] = 1.0
    for bot_id, bot in state.bots2.items():
        for part in bot.body:
            tensor[3][part.y][part.x] = 1.0
    return torch.tensor(tensor).unsqueeze(0)

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
    net = AlphaZeroNet(channels=4)
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    epochs = 10
    games_per_epoch = 10
    mcts_sims = 25

    for epoch in range(epochs):
        memory = []
        print(f"Epoch {epoch+1}/{epochs}")

        for game in range(games_per_epoch):
            state = snake_engine.GameState()
            game_history = []

            while len(state.bots1) > 0 and len(state.bots2) > 0:
                tensor_state = state_to_tensor(state)
                
                my_actions, my_visits = run_mcts(state, net, mcts_sims, True)
                opp_actions, opp_visits = run_mcts(state, net, mcts_sims, False)

                game_history.append(tensor_state)
                state.step(my_actions, opp_actions)

            reward = 0
            if len(state.bots1) > 0 and len(state.bots2) == 0:
                reward = 1
            elif len(state.bots2) > 0 and len(state.bots1) == 0:
                reward = -1

            for hist_state in game_history:
                memory.append((hist_state, reward))

        net.train()
        total_loss = 0

        for batch_state, batch_reward in memory:
            optimizer.zero_grad()
            _, pred_value = net(batch_state)
            target_value = torch.tensor([[batch_reward]], dtype=torch.float32)
            loss = nn.MSELoss()(pred_value, target_value)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Loss: {total_loss/len(memory) if memory else 0:.4f}")

if __name__ == "__main__":
    train_alphazero()