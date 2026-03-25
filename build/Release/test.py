import snake_engine

state = snake_engine.GameState()

print(state.width, state.height)

for bot_id, bot in state.bots1.items():
    print(bot_id, bot.body[0].x, bot.body[0].y)

my_actions = {0: "RIGHT", 2: "UP"}
opp_actions = {1: "LEFT", 3: "DOWN"}

state.step(my_actions, opp_actions)

for bot_id, bot in state.bots1.items():
    print(bot_id, bot.body[0].x, bot.body[0].y)
    
print(state.grid)