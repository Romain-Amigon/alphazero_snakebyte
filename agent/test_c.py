import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "..", "build", "Release"))
import snake_engine

state = snake_engine.GameState()
print("Init Tete 0:", state.bots1[0].body[0].x, state.bots1[0].body[0].y)
acts1 = {0: "LEFT", 2: "LEFT", 4: "LEFT", 6: "LEFT"}
acts2 = {1: "LEFT", 3: "LEFT", 5: "LEFT", 7: "LEFT"}
state.step(acts1, acts2)
print("Tour 1 Tete 0:", state.bots1[0].body[0].x, state.bots1[0].body[0].y)
