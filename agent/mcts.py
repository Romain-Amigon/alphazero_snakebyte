import math
import numpy as np
import torch
import torch.nn.functional as F

dirs = ["UP", "DOWN", "LEFT", "RIGHT"]

def get_tensor(state, is_p1, focus_bot_id=None):
    import snake_engine
    tensor = np.zeros((4, state.height, state.width), dtype=np.float32)
    
    for coord, tile in state.grid.cells.items():
        if tile.getType() == snake_engine.TileType.TYPE_WALL:
            tensor[0][coord.y][coord.x] = 1.0
            
    for apple in state.grid.apples:
        tensor[1][apple.y][apple.x] = 1.0
    
    my_bots = state.bots1 if is_p1 else state.bots2
    opp_bots = state.bots2 if is_p1 else state.bots1
    
    for bot_id, bot in my_bots.items():
        n = len(bot.body)
        for i, part in enumerate(bot.body):
            if 0 <= part.y < state.height and 0 <= part.x < state.width:
                val = 1.0 + (n - i) / max(1, n)
                # Si c'est le bot qu'on controle, la tete brille a 2.0. Sinon elle reste sous 1.5 max (comme un corps)
                if focus_bot_id is not None and bot_id == focus_bot_id:
                    tensor[2][part.y][part.x] = max(tensor[2][part.y][part.x], val)
                else:
                    # Les allies normaux sont vus comme des obstacles/corps amis (valeur max 1.0)
                    tensor[2][part.y][part.x] = max(tensor[2][part.y][part.x], 1.0)
                    
    for bot_id, bot in opp_bots.items():
        n = len(bot.body)
        for i, part in enumerate(bot.body):
            if 0 <= part.y < state.height and 0 <= part.x < state.width:
                tensor[3][part.y][part.x] = max(tensor[3][part.y][part.x], 1.0 + (n - i) / max(1, n))
            
    # Avoiding multiple GPU queries inside high freq loop
    device = next(torch.device("cuda" if torch.cuda.is_available() else "cpu") for _ in range(1))
    return torch.tensor(tensor).unsqueeze(0).to(device)

def get_valid_actions(state, is_p1):
    my_bots = state.bots1 if is_p1 else state.bots2
    valid = {}
    for bot_id, bot in my_bots.items():
        valid[bot_id] = dirs
    return valid

class MCTSTree:
    def __init__(self, net, is_p1):
        self.net = net
        self.is_p1 = is_p1
        self.Qsa = {}  
        self.Nsa = {}  
        self.Ns = {}   
        self.Ps = {}   
        self.Es = {}   

    def search(self, state, depth=0):
        my_bots = state.bots1 if self.is_p1 else state.bots2
        opp_bots = state.bots2 if self.is_p1 else state.bots1
        
        if len(my_bots) == 0 and len(opp_bots) == 0:
            return 0.0
        elif len(my_bots) == 0:
            return -1.0
        elif len(opp_bots) == 0:
            return 1.0
            
        if depth > 25: 
            score1 = sum(len(b.body) for b in my_bots.values())
            score2 = sum(len(b.body) for b in opp_bots.values())
            return 0.5 if score1 > score2 else (-0.5 if score2 > score1 else 0.0)
            
        s = str([(b.id, [(p.x, p.y) for p in b.body]) for b in my_bots.values()]) + "|" + str([(b.id, [(p.x, p.y) for p in b.body]) for b in opp_bots.values()])

        if s not in self.Ps:
            self.Ps[s] = {}
            total_v = 0.0
            
            for b_id in my_bots.keys():
                tensor_obs = get_tensor(state, self.is_p1, focus_bot_id=b_id)
                self.net.eval()
                with torch.no_grad():
                    pi, v = self.net(tensor_obs)
                
                pi = torch.softmax(pi, dim=1).cpu().numpy()[0]
                
                # Ajout du bruit de dirichlet à la racine
                if depth == 0:
                    noise = np.random.dirichlet([0.3] * len(pi))
                    pi = 0.75 * pi + 0.25 * noise
                
                self.Ps[s][b_id] = pi
                total_v += v.item()
                
            self.Ns[s] = 0
            # Return average value of the state based on all bots' perspectives
            return total_v / max(1, len(my_bots))

        cur_best = -float('inf')
        best_a_dict = {}
        
        # Build list of possible joint actions
        import itertools
        valid_acts = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        bot_ids = list(my_bots.keys())
        
        if len(bot_ids) == 0: return 0.0
        
        bot_act_combos = [dict(zip(bot_ids, acts)) for acts in itertools.product(valid_acts, repeat=len(bot_ids))]
        
        for a_dict in bot_act_combos:
            a_dict_str = str(a_dict)
            p_val = 1.0
            for b_id in bot_ids:
                p_val *= self.Ps[s][b_id][valid_acts.index(a_dict[b_id])]
            # On prend la moyenne géométrique pour éviter que la proba s'écrase vers 0 !
            p_val = p_val ** (1.0 / max(1, len(bot_ids)))
            
            c_puct = 2.0 # Force une forte exploration sur les noeuds inconnus
            
            if (s, a_dict_str) in self.Qsa:
                u = self.Qsa[(s, a_dict_str)] + c_puct * p_val * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a_dict_str)])
            else:
                u = c_puct * p_val * math.sqrt(self.Ns[s] + 1e-8)
                
            if u > cur_best:
                cur_best = u
                best_a_dict = a_dict
                
        a_dict = best_a_dict
        a_dict_str = str(a_dict)
        
        next_state = state.copy()
        import random
        opp_acts = {b: random.choice(dirs) for b in opp_bots.keys()}
        
        if self.is_p1:
            next_state.step(a_dict, opp_acts)
        else:
            next_state.step(opp_acts, a_dict)
            
        v = self.search(next_state, depth+1)
        
        if (s, a_dict_str) in self.Qsa:
            self.Qsa[(s, a_dict_str)] = (self.Nsa[(s, a_dict_str)] * self.Qsa[(s, a_dict_str)] + v) / (self.Nsa[(s, a_dict_str)] + 1)
            self.Nsa[(s, a_dict_str)] += 1
        else:
            self.Qsa[(s, a_dict_str)] = v
            self.Nsa[(s, a_dict_str)] = 1
            
        self.Ns[s] += 1
        return v


def run_mcts(state, net, mcts_sims, is_p1, tree=None):
    if tree is None:
        tree = MCTSTree(net, is_p1)
    for _ in range(mcts_sims):
        tree.search(state.copy())
        
    my_bots = state.bots1 if is_p1 else state.bots2
    if len(my_bots) == 0:
        return {}, {}, tree
        
    actions = {}
    visits = {}
    
    bot_ids = list(my_bots.keys())
    s = str([(b.id, [(p.x, p.y) for p in b.body]) for b in my_bots.values()]) + "|" + str([(b.id, [(p.x, p.y) for p in b.body]) for b in (state.bots2 if is_p1 else state.bots1).values()])
    
    import itertools
    valid_acts = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    bot_act_combos = [dict(zip(bot_ids, acts)) for acts in itertools.product(valid_acts, repeat=len(bot_ids))]
    
    best_combo = None
    max_v = -1
    
    for combo in bot_act_combos:
        combo_str = str(combo)
        v = tree.Nsa.get((s, combo_str), 0)
        visits[combo_str] = v
        if v > max_v:
            max_v = v
            best_combo = combo
            
    if best_combo is None:
        best_combo = {b_id: "UP" for b_id in bot_ids}
        
    return best_combo, visits, tree
