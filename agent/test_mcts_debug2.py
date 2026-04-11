import sys, os, torch
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "build", "Release"))
import snake_engine
from mcts import run_mcts
from train import AlphaZeroNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlphaZeroNet(channels=4).to(device)
state = snake_engine.GameState()
my_actions, visits, tree = run_mcts(state, net, mcts_sims=50, is_p1=True)

s2 = str([(b.id, [(p.x, p.y) for p in b.body]) for b in state.bots1.values()]) + "|" + str([(b.id, [(p.x, p.y) for p in b.body]) for b in state.bots2.values()])

import itertools
bot_ids = list(state.bots1.keys())
bot_act_combos = [dict(zip(bot_ids, acts)) for acts in itertools.product(['UP', 'DOWN', 'LEFT', 'RIGHT'], repeat=len(bot_ids))]

print("String root:", s2)
for combo in bot_act_combos:
    v = tree.Nsa.get((s2, str(combo)), 0)
    if v > 0:
        print("FOUND NSA:", combo, " -> ", v)
print("My_actions output:", my_actions)
