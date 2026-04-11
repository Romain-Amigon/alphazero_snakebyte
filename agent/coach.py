import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random

from mcts import MCTS
from resnet import AlphaZeroResNet
from config import Config

class Coach:
    def __init__(self, env, action_size=4):
        self.env = env
        self.action_size = action_size
        self.config = Config()
        
        w = self.env.state.width
        h = self.env.state.height
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"AlphaZero is running on: {self.device}")
        
        self.nnet = AlphaZeroResNet(board_width=w, board_height=h, channels=4, action_size=action_size).to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=self.config.LR)
        
    def execute_episode(self):
        train_examples = []
        current_env = self.env.clone()
        mcts = MCTS(self.nnet, self.config)
        step = 0
        
        while True:
            step += 1
            temp = int(step < self.config.TEMP_THRESHOLD)
            
            pi = mcts.getActionProb(current_env, temp=temp)
            sym = current_env.get_observation_tensor()
            train_examples.append([sym, current_env.is_player_1, pi, None])
            
            action = np.random.choice(len(pi), p=pi)
            current_env = current_env.take_action(action)
            
            r = current_env.get_reward()
            
            if r != 0 or current_env.is_terminal() or step >= self.config.MAX_TURNS:
                # terminal
                res = []
                for x in train_examples:
                    # If x[1] == current_env.is_player_1, the reward matches. Else inverse
                    actual_reward = r if x[1] == current_env.is_player_1 else -r
                    res.append((x[0], x[2], actual_reward))
                return res

    def learn(self):
        for i in range(1, self.config.EPOCHS + 1):
            print(f"Starting Epoch {i} ...")
            iteration_train_examples = deque([], maxlen=4000)
            
            for eps in range(self.config.SELF_PLAY_EPISODES):
                iteration_train_examples += self.execute_episode()
                
            self.train_network(iteration_train_examples)

    def train_network(self, examples):
        self.nnet.train()
        batch_size = self.config.BATCH_SIZE
        
        for epoch in range(10):
            print(f"NN Training Epoch {epoch + 1}/10")
            batch_count = int(len(examples) / batch_size)
            
            for _ in range(batch_count):
                sample = random.sample(examples, batch_size)
                
                states = torch.stack([s[0] for s in sample]).to(self.device)
                pis = torch.FloatTensor(np.array([s[1] for s in sample])).to(self.device)
                vs = torch.FloatTensor(np.array([s[2] for s in sample])).unsqueeze(1).to(self.device)
                
                out_pi, out_v = self.nnet(states)
                
                l_pi = -torch.sum(pis * out_pi) / pis.size()[0]
                l_v = torch.sum((vs - out_v) ** 2) / vs.size()[0]
                total_loss = l_pi + l_v
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
