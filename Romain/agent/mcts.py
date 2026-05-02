import math
import torch
import random
import numpy as np
from itertools import product
import snake_engine

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]
OPPOSITES = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
DIR_VECS = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}

class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.children = {}

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

def get_joint_actions(bots, state):
    bot_ids = list(bots.keys())
    valid_dirs = []
    
    for bot_id in bot_ids:
        bot = bots[bot_id]
        opp = OPPOSITES.get(bot.last_move, "")
        possible = []
        head = bot.body[0]
        
        for d in DIRS:
            if d == opp:
                continue
            
            nx = head.x + DIR_VECS[d][0]
            ny = head.y + DIR_VECS[d][1]
            
            if 0 <= nx < state.width and 0 <= ny < state.height:
                tile = state.grid.get(nx, ny)
                if tile.getType() == snake_engine.TileType.TYPE_WALL:
                    continue
            else:
                continue
                
            possible.append(d)
            
        if not possible:
            possible = [DIRS[0]]
            
        valid_dirs.append(possible)
        
    all_combinations = product(*valid_dirs)
    joint_actions = []
    
    for combo in all_combinations:
        action_dict = {bot_ids[i]: combo[i] for i in range(len(bot_ids))}
        joint_actions.append(action_dict)
        
    return joint_actions

def puct_score(child, parent_visit_count, c_puct=1.0):
    pb_c = c_puct * child.prior * math.sqrt(parent_visit_count) / (child.visit_count + 1)
    return child.value() + pb_c

def select_child(node):
    best_score = -float('inf')
    best_action = None
    best_child = None
    for action_key, (ja, child) in node.children.items():
        score = puct_score(child, node.visit_count)
        if score > best_score:
            best_score = score
            best_action = ja
            best_child = child
    return best_action, best_child

def run_mcts(state, net, num_simulations, is_player_1=True, temperature=1.0, add_noise=False):
    root = Node(1.0)
    
    my_bots_init = state.bots1 if is_player_1 else state.bots2
    if len(my_bots_init) == 0:
        return {}, []
        
    for _ in range(num_simulations):
        node = root
        sim_state = state.copy()
        search_path = [node]

        while node.expanded():
            my_action_dict, node = select_child(node)
            
            opp_action_dict = {}
            opp_bots = sim_state.bots2 if is_player_1 else sim_state.bots1
            for opp_id, opp_bot in opp_bots.items():
                opp_opp = OPPOSITES.get(opp_bot.last_move, "")
                opp_possible = [d for d in DIRS if d != opp_opp]
                opp_action_dict[opp_id] = random.choice(opp_possible) if opp_possible else DIRS[0]

            if is_player_1:
                sim_state.step(my_action_dict, opp_action_dict)
            else:
                sim_state.step(opp_action_dict, my_action_dict)
                
            search_path.append(node)

        value = 0
        is_terminal = len(sim_state.bots1) == 0 or len(sim_state.bots2) == 0
        
        if is_terminal:
            if len(sim_state.bots1) > 0:
                value = 1 if is_player_1 else -1
            elif len(sim_state.bots2) > 0:
                value = -1 if is_player_1 else 1
            else:
                value = 0
        else:
            from train import state_to_tensor
            tensor_state = state_to_tensor(sim_state)
            with torch.no_grad():
                policy_tensor, value_tensor = net(tensor_state)
            value = value_tensor.item()
            
            probs = torch.softmax(policy_tensor[0], dim=0).cpu().numpy()
            dir_probs = {DIRS[i]: probs[i] for i in range(4)}
            
            my_bots = sim_state.bots1 if is_player_1 else sim_state.bots2
            joint_actions = get_joint_actions(my_bots, sim_state)
            
            action_priors = []
            for ja in joint_actions:
                p = 1.0
                for bot_id, d in ja.items():
                    p *= dir_probs[d]
                action_priors.append(p)
                
            sum_priors = sum(action_priors)
            if sum_priors > 0:
                action_priors = [p / sum_priors for p in action_priors]
            else:
                action_priors = [1.0 / len(joint_actions)] * len(joint_actions)
            
            for ja, p in zip(joint_actions, action_priors):
                action_key = tuple(sorted(ja.items()))
                node.children[action_key] = (ja, Node(p))

            if node == root and add_noise:
                noise = np.random.dirichlet([0.3] * len(action_priors))
                for i, (action_key, (ja, child)) in enumerate(node.children.items()):
                    child.prior = 0.75 * child.prior + 0.25 * noise[i]

        for n in reversed(search_path):
            n.value_sum += value
            n.visit_count += 1
            value = -value

    action_visits = []
    counts = []
    for action_key, (ja, child) in root.children.items():
        action_visits.append(ja)
        counts.append(child.visit_count)
        
    if sum(counts) > 0:
        if temperature == 0:
            chosen_action = action_visits[np.argmax(counts)]
        else:
            probs = [c / sum(counts) for c in counts]
            idx = np.random.choice(len(action_visits), p=probs)
            chosen_action = action_visits[idx]
    else:
        chosen_action = random.choice(action_visits) if action_visits else {}
        
    visits_out = [(ja, count) for ja, count in zip(action_visits, counts)]
    return chosen_action, visits_out