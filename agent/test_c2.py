import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "build", "Release"))
import snake_engine

state = snake_engine.GameState()
print("Init Tete 0:", state.bots1[0].body[0].x, state.bots1[0].body[0].y)
print("Body 0:", [(p.x, p.y) for p in state.bots1[0].body])
acts1 = {0: "UP"}
acts2 = {}
state.step(acts1, acts2)
if 0 in state.bots1:
    print("Tour 1 Tete 0:", state.bots1[0].body[0].x, state.bots1[0].body[0].y)
else:
    print("Bot 0 est mort !")
