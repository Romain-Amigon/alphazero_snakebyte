import gymnasium as gym
from gymnasium import spaces
import numpy as np
import collections

try:
    import snake_engine
except ImportError:
    import sys, os
    # Tentative d'import local si l'engine n'est pas dans le PATH
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(current_dir, "..", "build", "Release"))
    import snake_engine

DIRS = ["UP", "DOWN", "LEFT", "RIGHT"]
INT_TO_DIR = {i: d for i, d in enumerate(DIRS)}
OPPOSITE_ACTION = {0: 1, 1: 0, 2: 3, 3: 2}  # UP<->DOWN, LEFT<->RIGHT
DELTA = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}

class SnakePPOEnv(gym.Env):
    """
    Environnement compatible avec Gymnasium pour entraîner l'agent avec PPO
    via Stable-Baselines3.
    """
    metadata = {'render_modes': ['console']}
    
    def __init__(self, max_steps=500):
        super(SnakePPOEnv, self).__init__()
        
        # PPO va contrôler 1 seul et unique serpent (le chef, le leader).
        self.action_space = spaces.Discrete(4)
        
        # On définit une taille fixe et suffisamment grande pour l'espace d'observation de Gym
        # puisque ton GameState C++ a l'air de générer des cartes de tailles variables (ex: 40x22, 38x...).
        self.max_width = 50
        self.max_height = 50
        
        # Observation : Dictionnaire (image + vecteur)
        # L'image est en format channel-first (C, H, W) pour Conv2d
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0, high=255,
                shape=(5, self.max_height, self.max_width),
                dtype=np.uint8
            ),
            # Vecteur de caractéristiques (6 valeurs) : 
            # [dx_pomme, dy_pomme, mur_haut, mur_bas, mur_gauche, mur_droit]
            "vector": spaces.Box(
                low=-1.0, high=1.0, 
                shape=(6,), 
                dtype=np.float32
            )
        })
        
        self.state = None
        self.focus_bot_id = None

        self.max_steps = max_steps
        self.current_step = 0

        # Variables de suivi pour les récompenses personnalisées
        self.last_score = 0
        self.last_action = None
        self.last_head_pos = None
        self.last_dist_to_apple = None
        # Fenêtre courte adaptée à des parties de ~30 tours : détecte les
        # petites boucles (2-8 cases) sans être dilué par l'historique ancien
        self.visited_positions = collections.deque(maxlen=15)
        self.stuck_counter = 0
        self._obstacle_cache = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = snake_engine.GameState()
        
        # S'assure que le plateau actuel ne dépasse pas nos bornes (sinon il faudra l'agrandir).
        assert self.state.width <= self.max_width, f"Width {self.state.width} exceeds MAX {self.max_width}"
        assert self.state.height <= self.max_height, f"Height {self.state.height} exceeds MAX {self.max_height}"
        
        self.current_step = 0
        self.last_action = None
        self.visited_positions.clear()
        self.last_dist_to_apple = None
        self.stuck_counter = 0

        # On choisit aléatoirement un des serpents vivants de l'équipe 1 comme le "cerveau" courant
        alive_bots = list(self.state.bots1.keys())
        if alive_bots:
            self.focus_bot_id = np.random.choice(alive_bots)
            bot = self.state.bots1[self.focus_bot_id]
            self.last_head_pos = (bot.body[0].x, bot.body[0].y)
            # FIX: initialiser last_score avec la vraie taille du corps (3 au spawn)
            # au lieu de 0, sinon la première "pomme" détectée est un faux positif.
            self.last_score = len(bot.body)
        else:
            self.focus_bot_id = None
            self.last_head_pos = None
            self.last_score = 0

        return self._get_obs(self.focus_bot_id), {}

    def _get_my_bot_id(self):
        # On retourne maintenant l'ID spécifique focalisé !
        return self.focus_bot_id

    def _get_opp_bot_id(self):
        return list(self.state.bots2.keys())[0] if self.state.bots2 else None

    def step(self, action):
        self.current_step += 1

        my_bot_id = self._get_my_bot_id()

        if my_bot_id is None or my_bot_id not in self.state.bots1:
            return self._get_obs(my_bot_id), -1.0, True, False, {"reason": "already_dead"}

        focus_bot = self.state.bots1[my_bot_id]
        head = focus_bot.body[0]

        # ============================================================
        # Pré-pénalité unique : action manifestement suicidaire
        # (mur OU son propre corps). On retire l'opposite-action et on
        # baisse l'intensité — sinon le bot a peur d'explorer et meurt
        # en 12 coups au lieu d'apprendre.
        # ============================================================
        dx, dy = DELTA[action]
        next_head = (head.x + dx, head.y + dy)
        suicide = False
        if (next_head[0] < 0 or next_head[0] >= self.state.width or
            next_head[1] < 0 or next_head[1] >= self.state.height):
            suicide = True
        else:
            own_body_coords = {(c.x, c.y) for c in list(focus_bot.body)[:-1]}
            if next_head in own_body_coords:
                suicide = True
            elif self.state.grid.get(next_head[0], next_head[1]).getType() == 1:
                suicide = True
        pre_penalty = -0.15 if suicide else 0.0

        action_str = INT_TO_DIR[action]

        # Cache des obstacles UNE seule fois par step (gros gain CPU)
        self._obstacle_cache = self._build_obstacle_set()

        my_actions = {my_bot_id: action_str}
        for b_id in self.state.bots1.keys():
            if b_id != my_bot_id:
                my_actions[b_id] = self._get_smart_team_action(b_id, team=1)

        opp_actions = {}
        for b_id in self.state.bots2.keys():
            opp_actions[b_id] = self._get_smart_team_action(b_id, team=2)

        self.state.step(my_actions, opp_actions)
        self._obstacle_cache = None  # invalidé après step

        # ================================
        # RÉCOMPENSES POST-STEP
        # ================================
        reward = pre_penalty
        terminated = False

        # 1. Survie : doit dominer la pénalité de boucle pour éviter
        # que l'agent apprenne à se suicider plutôt que de tourner en rond
        reward += 0.01

        new_my_bot_id = self._get_my_bot_id()

        # 2. Mort
        if new_my_bot_id is None or new_my_bot_id not in self.state.bots1:
            reward -= 3.0
            terminated = True
        else:
            bot = self.state.bots1[new_my_bot_id]
            current_score = len(bot.body)
            new_head_pos = (bot.body[0].x, bot.body[0].y)

            # 3. Pomme : signal dominant
            if current_score > self.last_score:
                reward += 4.0
                self.visited_positions.clear()
                self.stuck_counter = 0
            self.last_score = current_score

            # 4. Distance (potential-based, faible)
            if self.state.grid.apples:
                curr_dist = min(
                    abs(a.x - new_head_pos[0]) + abs(a.y - new_head_pos[1])
                    for a in self.state.grid.apples
                )
                if self.last_dist_to_apple is not None:
                    delta = max(-2, min(2, self.last_dist_to_apple - curr_dist))
                    reward += 0.02 * delta
                self.last_dist_to_apple = curr_dist
            else:
                self.last_dist_to_apple = None

            # 5. Boucle : pénalité fixe simple (pas d'escalade)
            if new_head_pos in self.visited_positions:
                reward -= 0.02
            self.visited_positions.append(new_head_pos)

            # 6. Immobilité : CAP À 3 pour éviter explosion en eval
            # (avant : -0.2 * 500 = -100 par step → -2355 en eval)
            if self.last_head_pos == new_head_pos:
                self.stuck_counter = min(self.stuck_counter + 1, 3)
                reward -= 0.1 * self.stuck_counter
            else:
                self.stuck_counter = 0
            self.last_head_pos = new_head_pos

        # 7. Victoire
        if len(self.state.bots2) == 0:
            reward += 3.0
            terminated = True

        self.last_action = action
        truncated = self.current_step >= self.max_steps

        return self._get_obs(my_bot_id), reward, terminated, truncated, {}

    def _get_smart_opp_action(self, opp_id):
        # Conservé pour compat externe (test_ppo.py). Délègue à la version générique.
        return self._get_smart_team_action(opp_id, team=2)

    def _build_obstacle_set(self):
        """Construit l'ensemble murs + tous les corps une fois par step."""
        obstacles = set()
        W, H = self.state.width, self.state.height
        grid_get = self.state.grid.get
        for y in range(H):
            for x in range(W):
                if grid_get(x, y).getType() == 1:
                    obstacles.add((x, y))
        for b in self.state.bots1.values():
            for c in b.body:
                obstacles.add((c.x, c.y))
        for b in self.state.bots2.values():
            for c in b.body:
                obstacles.add((c.x, c.y))
        return obstacles

    def _get_smart_team_action(self, bot_id, team=2):
        """BFS vers la pomme la plus proche pour un bot de team 1 ou 2."""
        bots_self = self.state.bots1 if team == 1 else self.state.bots2
        bot = bots_self.get(bot_id)
        if not bot or len(bot.body) == 0:
            return np.random.choice(DIRS)

        start = (bot.body[0].x, bot.body[0].y)
        targets = set((a.x, a.y) for a in self.state.grid.apples)

        if not targets:
            return np.random.choice(DIRS)

        # Réutilise le cache calculé dans step() si dispo (gros gain perf)
        cache = getattr(self, "_obstacle_cache", None)
        obstacles = set(cache) if cache is not None else self._build_obstacle_set()
        obstacles.discard(start)
            
        queue = collections.deque([(start, [])])
        visited = set([start])
        
        # Directions standards sur grille: (Y négatif = UP, Y positif = DOWN)
        dirs = {(0, -1): "UP", (0, 1): "DOWN", (-1, 0): "LEFT", (1, 0): "RIGHT"}
        
        while queue:
            curr, path = queue.popleft()
            
            # S'il a atteint une cible, on retourne le premeir mouvement de ce chemin
            if curr in targets:
                return path[0] if path else np.random.choice(DIRS)
                
            cx, cy = curr
            for (dx, dy), dname in dirs.items():
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.state.width and 0 <= ny < self.state.height:
                    if (nx, ny) not in obstacles and (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append(((nx, ny), path + [dname]))
                        
        # Si aucun chemin n'a été trouvé vers une pomme (bloqué), 
        # on choisit la première case libre pour survivre un tour de plus
        for (dx, dy), dname in dirs.items():
            nx, ny = start[0] + dx, start[1] + dy
            if 0 <= nx < self.state.width and 0 <= ny < self.state.height:
                if (nx, ny) not in obstacles:
                    return dname
                    
        # Aucun mouvement possible, on s'écrase
        return np.random.choice(DIRS)

    def _get_obs(self, override_focus_id=None):
        focus_id = override_focus_id if override_focus_id is not None else self.focus_bot_id
        # Canaux : 0=Murs, 1=Moi, 2=Ennemis, 3=Pommes, 4=Alliés — format (C, H, W)
        tensor = np.zeros((5, self.max_height, self.max_width), dtype=np.uint8)

        # Padding : zones hors carte → canal murs
        tensor[0, self.state.height:, :] = 255
        tensor[0, :, self.state.width:] = 255

        obstacles = set()
        for y in range(self.state.height):
            for x in range(self.state.width):
                if self.state.grid.get(x, y).getType() == 1:
                    tensor[0, y, x] = 255
                    obstacles.add((x, y))

        for apple in self.state.grid.apples:
            if 0 <= apple.x < self.state.width and 0 <= apple.y < self.state.height:
                tensor[3, apple.y, apple.x] = 255

        # Alliés (canal 1 = moi, canal 4 = alliés)
        head_x, head_y = -1, -1
        for b_id, bot in self.state.bots1.items():
            length = len(bot.body)
            if length == 0:
                continue

            channel = 1 if b_id == focus_id else 4

            for i, c in enumerate(bot.body):
                obstacles.add((c.x, c.y))
                if i == 0 and b_id == focus_id:
                    head_x, head_y = c.x, c.y

                if 0 <= c.x < self.max_width and 0 <= c.y < self.max_height:
                    intensity = int(255 - (i / max(1, length)) * 200)
                    tensor[channel, c.y, c.x] = max(50, intensity)

        # Ennemis (canal 2)
        for b_id, bot in self.state.bots2.items():
            length = len(bot.body)
            if length == 0:
                continue

            for i, c in enumerate(bot.body):
                obstacles.add((c.x, c.y))
                if 0 <= c.x < self.max_width and 0 <= c.y < self.max_height:
                    intensity = int(255 - (i / max(1, length)) * 200)
                    tensor[2, c.y, c.x] = max(50, intensity)

        # --- VECTEUR EXPLICITE ---
        if head_x != -1 and head_y != -1:
            min_dist = 9999
            apple_dx, apple_dy = 0.0, 0.0
            for apple in self.state.grid.apples:
                dist = abs(apple.x - head_x) + abs(apple.y - head_y)
                if dist < min_dist:
                    min_dist = dist
                    apple_dx = (apple.x - head_x) / self.max_width
                    apple_dy = (apple.y - head_y) / self.max_height

            def is_obs(x, y):
                if x < 0 or x >= self.state.width or y < 0 or y >= self.state.height:
                    return 1.0
                return 1.0 if (x, y) in obstacles else 0.0

            vector = np.array([
                apple_dx, apple_dy,
                is_obs(head_x, head_y - 1),
                is_obs(head_x, head_y + 1),
                is_obs(head_x - 1, head_y),
                is_obs(head_x + 1, head_y),
            ], dtype=np.float32)
        else:
            vector = np.zeros(6, dtype=np.float32)

        return {"image": tensor, "vector": vector}
