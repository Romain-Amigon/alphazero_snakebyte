import math
p_val = (0.25)**4  # 4 bots
Ns = 1
Qsa = 0.05
# Visited
u_visited = Qsa + 1.0 * p_val * math.sqrt(Ns) / (1 + 1)
# Unvisited
u_unvisited = 1.0 * p_val * math.sqrt(Ns + 1e-8)

print("Visited PUCT:", u_visited)
print("Unvisited PUCT:", u_unvisited)
