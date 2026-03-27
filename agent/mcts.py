import math
import torch
import random
import numpy as np
from itertools import product

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]

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

def get_joint_actions(bots):
    bot_ids = list(bots.keys())
    all_combinations = product(DIRS, repeat=len(bot_ids))
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
    for action_str, child in node.children.items():
        score = puct_score(child, node.visit_count)
        if score > best_score:
            best_score = score
            best_action = action_str
            best_child = child
    return best_action, best_child

def run_mcts(state, net, num_simulations, is_player_1=True):
    root = Node(1.0)
    
    my_bots_init = state.bots1 if is_player_1 else state.bots2
    if len(my_bots_init) == 0:
        return {}, {}
        
    for _ in range(num_simulations):
        node = root
        sim_state = state.copy()
        search_path = [node]

        while node.expanded():
            action_str, node = select_child(node)
            my_action_dict = eval(action_str)
            
            opp_action_dict = {}
            opp_bots = sim_state.bots2 if is_player_1 else sim_state.bots1
            for opp_id in opp_bots.keys():
                opp_action_dict[opp_id] = random.choice(DIRS)

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
            joint_actions = get_joint_actions(my_bots)
            
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
                node.children[str(ja)] = Node(p)

        for n in reversed(search_path):
            n.value_sum += value
            n.visit_count += 1
            value = -value

    action_visits = {}
    for action_str, child in root.children.items():
        action_visits[action_str] = child.visit_count
        
    actions = list(action_visits.keys())
    counts = list(action_visits.values())
    
    if sum(counts) > 0:
        probs = [c / sum(counts) for c in counts]
        chosen_action_str = np.random.choice(actions, p=probs)
    else:
        chosen_action_str = random.choice(actions) if actions else "{}"
        
    return eval(chosen_action_str), action_visits