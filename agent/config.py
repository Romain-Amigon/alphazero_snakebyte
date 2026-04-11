class Config:
    NUM_SIMULATIONS = 25
    C_PUCT = 1.0
    LR = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 10
    SELF_PLAY_EPISODES = 50
    ARENA_GAMES = 20
    UPDATE_THRESHOLD = 0.55
    TEMP_THRESHOLD = 15 # Temperature threshold
    MAX_TURNS = 200    # Prevent infinite games
