import sys, os, torch
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)
import snake_engine
from mcts import run_mcts
from train import AlphaZeroNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlphaZeroNet(channels=4).to(device)
state = snake_engine.GameState()
my_actions, visits, tree = run_mcts(state, net, mcts_sims=50, is_p1=True)

s = str([(b.id, [(p.x, p.y) for p in b.body]) for b in state.bots1.values()]) + "|" + str([(b.id, [(p.x, p.y) for p in state.bots2.values()]) for b in state.bots2.values()])
s = str([(b.id, [(p.x, p.y) for p in b.body]) for b in state.bots1.values()]) + "|" + str([(b.id, [(p.x, p.y) for p in b.body]) for b in state.bots2.values()])

import itertools
bot_ids = list(state.bots1.keys())
valid_acts = ['UP', 'DOWN', 'LEFT', 'RIGHT']
bot_act_combos = [dict(zip(bot_ids, acts)) for acts in itertools.product(valid_acts, repeat=len(bot_ids))]

print("String root:", s)
found_any = False
for combo in bot_act_combos:
    v = tree.Nsa.get((s, str(combo)), 0)
    if v > 0:
        found_any = True
        print("FOUND!", combo, " -> ", v)
print("Was any positive Nsa found for root?", found_any)
