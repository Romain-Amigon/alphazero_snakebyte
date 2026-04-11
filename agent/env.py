import numpy as np
import torch
import copy
import random

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]
DIR_TO_INT = {d: i for i, d in enumerate(DIRS)}
INT_TO_DIR = {i: d for i, d in enumerate(DIRS)}

class SnakeEnv:
    def __init__(self, c_state, is_player_1=True):
        self.state = c_state
        self.is_player_1 = is_player_1
        self.action_size = 4
        
    def clone(self):
        # GameState::copy() is exposed via pybind
        return SnakeEnv(self.state.copy(), self.is_player_1)
        
    def to_string(self):
        # A simple hashable representation of the state (from the tensor)
        # This is used by MCTS to cache node evaluations
        obs = self.get_observation_tensor()
        return obs.cpu().numpy().tobytes()
        
    def get_valid_actions(self):
        # Practically, all 4 moves might be "valid" in the sense that they can be sent to step(), 
        # but 180 degree turns (e.g., UP if last was DOWN) lead to instant self-collision and death.
        # However, to let the network learn everything or restrict it:
        # We will allow all actions, the network will quickly learn to avoid 180 turns.
        return [0, 1, 2, 3]
        
    def get_reward(self):
        # We need to know who won. If a bot dies, its body size is zero or it's removed?
        # Let's check by the length of the map of bots
        alive_p1 = len(self.state.bots1) > 0
        alive_p2 = len(self.state.bots2) > 0
        
        if alive_p1 and not alive_p2:
            return 1 if self.is_player_1 else -1    # p1 won
        elif alive_p2 and not alive_p1:
            return -1 if self.is_player_1 else 1    # p2 won
        elif not alive_p1 and not alive_p2:
            return 0                                # draw
        else:
            return 0                                # ongoing
            
    def is_terminal(self):
        return len(self.state.bots1) == 0 or len(self.state.bots2) == 0
        
    def take_action(self, action_int):
        """
        Since it's simultaneous, if we call take_action for P1, we need an opponent action.
        For self-play alpha zero with simultaneous turns, you can either pick a random opp action
        or a network predicted one. Here we will do a random action for the dummy step in MCTS, 
        or we alternate turns by swapping perspective.
        Proper way for simultaneous: we actually step the environment combining both actions.
        For strictly alternating MCTS: we queue our action, then swap perspective.
        """
        action_str = INT_TO_DIR[action_int]
        my_actions = {}
        opp_actions = {}
        
        # Determine IDs
        if self.is_player_1:
            # We are player 1, our bot is in bots1, opponent is in bots2
            bot_id = list(self.state.bots1.keys())[0] if self.state.bots1 else 0
            opp_id = list(self.state.bots2.keys())[0] if self.state.bots2 else 1
            
            my_actions[bot_id] = action_str
            opp_actions[opp_id] = random.choice(DIRS)
            
            next_c_state = self.state.copy()
            next_c_state.step(my_actions, opp_actions)
        else:
            # We are player 2, our bot is in bots2, opponent is in bots1
            bot_id = list(self.state.bots2.keys())[0] if self.state.bots2 else 1
            opp_id = list(self.state.bots1.keys())[0] if self.state.bots1 else 0
            
            my_actions[bot_id] = action_str
            opp_actions[opp_id] = random.choice(DIRS)
            
            next_c_state = self.state.copy()
            # C++ engine step expects (movesP1, movesP2)
            # Since opponent is P1, their moves go first
            next_c_state.step(opp_actions, my_actions)
            
        # Return new state wrapper, swapped perspective
        return SnakeEnv(next_c_state, not self.is_player_1)

    def get_observation_tensor(self):
        w = self.state.width
        h = self.state.height
        
        # 4 channels: Walls, My Body, Enemy Body, Apples
        tensor = np.zeros((4, h, w), dtype=np.float32)
        
        # Murs / Walls
        # Accessing cells from the Grid
        # However, grid.cells is a 1D vector of length w*h
        for y in range(h):
            for x in range(w):
                # We can use grid.get(x,y).getType()
                if self.state.grid.get(x, y).getType() == 1: # TYPE_WALL might be 1
                    tensor[0, y, x] = 1.0
                    
        # Apples
        for apple in self.state.grid.apples:
            tensor[3, apple.y, apple.x] = 1.0
            
        # Bodies
        my_bots = self.state.bots1 if self.is_player_1 else self.state.bots2
        opp_bots = self.state.bots2 if self.is_player_1 else self.state.bots1
        
        for b_id, bot in my_bots.items():
            for c in bot.body:
                if 0 <= c.x < w and 0 <= c.y < h:
                    tensor[1, c.y, c.x] = 1.0
                    
        for b_id, bot in opp_bots.items():
            for c in bot.body:
                if 0 <= c.x < w and 0 <= c.y < h:
                    tensor[2, c.y, c.x] = 1.0
                    
        return torch.FloatTensor(tensor)
