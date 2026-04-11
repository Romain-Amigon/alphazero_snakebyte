import sys, os, torch
current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir, "..", "build", "Release")
sys.path.append(build_dir)
import snake_engine
from mcts import run_mcts
from train import AlphaZeroNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = AlphaZeroNet(channels=4).to(device)
net.load_state_dict(torch.load("models/alphazero_snake_ep14.pth", map_location=device))
net.eval()

state = snake_engine.GameState()
my_tree = None
for tour in range(10):
    if len(state.bots1) > 0:
        for b_id, b in state.bots1.items():
            print(f"Tour {tour} Bleu {b_id} tete: ({b.body[0].x}, {b.body[0].y}) tail: ({b.body[-1].x}, {b.body[-1].y})")
    if len(state.bots2) > 0:
        for b_id, b in state.bots2.items():
            print(f"Tour {tour} Rouge {b_id} tete: ({b.body[0].x}, {b.body[0].y}) tail: ({b.body[-1].x}, {b.body[-1].y})")

    my_actions, _, my_tree = run_mcts(state, net, mcts_sims=50, is_p1=True, tree=my_tree)
    print("Action MCTS Bleu:", my_actions)
    import random_bot
    opp_actions = random_bot.run(state, is_player_1=False)
    state.step(my_actions, opp_actions)
