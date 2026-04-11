import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
from collections import deque
import random

current_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(current_dir,"..", "build", "Release")
sys.path.append(build_dir)
import snake_engine

from mcts import run_mcts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def state_to_tensor(state, local_device=device, focus_bot_id=None):
    tensor = np.zeros((4, state.height, state.width), dtype=np.float32)
    for coord, tile in state.grid.cells.items():
        if tile.getType() == snake_engine.TileType.TYPE_WALL:
            tensor[0][coord.y][coord.x] = 1.0
    for coord in state.grid.apples:
        tensor[1][coord.y][coord.x] = 1.0
    for bot_id, bot in state.bots1.items():
        n = len(bot.body)
        for i, part in enumerate(bot.body):
            if 0 <= part.y < state.height and 0 <= part.x < state.width:
                val = 1.0 + (n - i) / max(1, n)
                if focus_bot_id is not None and bot_id == focus_bot_id:
                    tensor[2][part.y][part.x] = max(tensor[2][part.y][part.x], val)
                else:
                    tensor[2][part.y][part.x] = max(tensor[2][part.y][part.x], 1.0)
    for bot_id, bot in state.bots2.items():
        n = len(bot.body)
        for i, part in enumerate(bot.body):
            if 0 <= part.y < state.height and 0 <= part.x < state.width:
                tensor[3][part.y][part.x] = max(tensor[3][part.y][part.x], 1.0 + (n - i) / max(1, n))
    return torch.tensor(tensor).unsqueeze(0).to(local_device)

class AlphaZeroNet(nn.Module):
    def __init__(self, channels):
        super(AlphaZeroNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten()
        )
        fc_size = 64 * 8 * 8
        self.policy_head = nn.Sequential(
            nn.Linear(fc_size, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )
        self.value_head = nn.Sequential(
            nn.Linear(fc_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.conv(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

class RemoteNet:
    def __init__(self, worker_id, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue

    def __call__(self, tensor):
        # Send to GPU inference server
        self.request_queue.put((self.worker_id, tensor.cpu()))
        # Wait for the batch response
        pi, v = self.response_queue.get()
        # MCTS code expects results on the same device it would run, but we can stick to returning values directly to CPU to avoid issues.
        device = next(torch.device("cuda" if torch.cuda.is_available() else "cpu") for _ in range(1))
        return pi.to(device), v.to(device)

    def eval(self):
        pass

def gpu_worker_loop(net, request_queue, response_queues):
    local_device = next(net.parameters()).device
    net.eval()
    print("GPU Worker Démarré.")
    while True:
        req = request_queue.get()
        if req == "STOP":
            break
        requests = [req]
        
        # Aspire jusqu'a 128 requetes en attente pour les batcher
        while not request_queue.empty() and len(requests) < 128:
            try:
                requests.append(request_queue.get_nowait())
            except:
                break
                
        # On concatene les tensors (et on pad avec F.pad car la taille du terrain peut varier)
        tensors = [r[1] for r in requests]
        
        max_h = max(t.shape[2] for t in tensors)
        max_w = max(t.shape[3] for t in tensors)
        padded_tensors = []
        for t in tensors:
            pad_h = max_h - t.shape[2]
            pad_w = max_w - t.shape[3]
            padded_tensors.append(F.pad(t, (0, pad_w, 0, pad_h)))
            
        batch = torch.cat(padded_tensors, dim=0).to(local_device)
        
        with torch.no_grad():
            batch_pi, batch_v = net(batch)
            
        batch_pi = batch_pi.cpu()
        batch_v = batch_v.cpu()
        
        for i, r in enumerate(requests):
            worker_id = r[0]
            # Chaque MCTS s'attend à recevoir [1, 4] et [1, 1], nous lisons donc des slices
            response_queues[worker_id].put((batch_pi[i:i+1], batch_v[i:i+1]))

def play_game(worker_id, request_queue, response_queue, mcts_sims):
    # RemoteNet remplace le vrai réseau pour le MCTS
    net = RemoteNet(worker_id, request_queue, response_queue)
    local_device = torch.device("cpu") # Tous les tensors initiaux resteront sur CPU
    
    state = snake_engine.GameState()
    game_history = []
    tour = 0
    dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
    my_tree = None
    opp_tree = None

    import math
    while len(state.bots1) > 0 and len(state.bots2) > 0 and tour < 200 and len(state.grid.apples) > 0:
        score1 = sum(len(bot.body) for bot in state.bots1.values())
        score2 = sum(len(bot.body) for bot in state.bots2.values())
        bot_count1 = len(state.bots1)
        
        my_actions, my_visits, my_tree = run_mcts(state, net, mcts_sims, True, my_tree)
        opp_actions, opp_visits, opp_tree = run_mcts(state, net, mcts_sims, False, opp_tree)

        bot_states = {}
        total_visits = sum(my_visits.values())
        if total_visits > 0:
            for b_id in state.bots1.keys():
                pi = np.zeros(4, dtype=np.float32)
                for combo_str, count in my_visits.items():
                    try:
                        combo_dict = eval(combo_str)
                        if b_id in combo_dict:
                            pi[dirs.index(combo_dict[b_id])] += count
                    except:
                        pass
                if np.sum(pi) > 0:
                    pi /= np.sum(pi)
                    
                bot_tensor = state_to_tensor(state, local_device, focus_bot_id=b_id).cpu().numpy()
                bot_states[b_id] = (bot_tensor, pi)
        
        ref_tensor = state_to_tensor(state, local_device, focus_bot_id=None).cpu().numpy()
        game_history.append({"bots_data": bot_states, "score1": score1, "score2": score2, "bots1": bot_count1, "ref_state": ref_tensor})
        
        state.step(my_actions, opp_actions)
        tour += 1

    final_score1 = sum(len(bot.body) for bot in state.bots1.values())
    final_score2 = sum(len(bot.body) for bot in state.bots2.values())

    result_data = []
    win_reward = 0.0
    if len(state.bots1) > 0 and len(state.bots2) == 0:
        win_reward = 1.0
    elif len(state.bots2) > 0 and len(state.bots1) == 0:
        win_reward = -1.0
    else:
        diff = final_score1 - final_score2
        win_reward = math.tanh(diff / 5.0)

    for i, step_data in enumerate(game_history):
        step_reward = 0.01 
        
        if i + 1 < len(game_history):
            next_step = game_history[i+1]
            if next_step["score1"] > step_data["score1"]:
                step_reward += 0.5
            if next_step["bots1"] < step_data["bots1"]:
                step_reward -= 0.5
            if np.array_equal(next_step["ref_state"][0][2], step_data["ref_state"][0][2]):
                step_reward -= 0.2
                
        discount = 0.95 ** (len(game_history) - i - 1)
        total_reward = win_reward * discount + step_reward
        total_reward = max(min(total_reward, 1.0), -1.0)
        
        for b_id, (tensor_st, pi) in step_data["bots_data"].items():
            result_data.append((tensor_st, pi, total_reward))
    return result_data


def train_alphazero():
    # Fixe le probleme de multiprocessing avec CUDA (souvent besoin de spawn)
    mp.set_start_method('spawn', force=True)
    
    print(f"Execution sur : {device}")
    
    # Net partagé entre les processus: on le garde d'abord sur l'hôte / GPU
    net = AlphaZeroNet(channels=4).to(device)
    net.share_memory() # Permet au multiprocessing de partager les tensors
    
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    criterion_value = nn.MSELoss()
    log_softmax = nn.LogSoftmax(dim=1)

    epochs = 500
    games_per_epoch = 100 # On joue 100 parties par époque
    mcts_sims = 50
    batch_size = 128 # Batch plus gros
    memory = deque(maxlen=20000)
    
    # On pousse à 50 processus en parallèle au lieu de 10 (Attention au CPU !)
    num_processes = 50 
    print(f"Lancement de {num_processes} parties en parallèle pour saturer l'architecture...")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        manager = mp.Manager()
        request_queue = manager.Queue()
        response_queues = [manager.Queue() for _ in range(num_processes)]
        
        # Start GPU Worker
        gpu_process = mp.Process(target=gpu_worker_loop, args=(net, request_queue, response_queues))
        gpu_process.start()

        pool = mp.Pool(processes=num_processes)
        
        # Async run games
        results = []
        for j in range(games_per_epoch):
            # Envoie shared net
            r = pool.apply_async(play_game, args=(j % num_processes, request_queue, response_queues[j % num_processes], mcts_sims))
            results.append(r)
            
        # Collecter les résultats
        for i, r in enumerate(results):
            game_data = r.get() # Attend la fin du jeu et récupère son memory
            memory.extend(game_data)
            print(f"\r  Partie {i+1}/{games_per_epoch} terminée", end="")
            sys.stdout.flush()
            
        pool.close()
        pool.join()

        # Stop GPU after all workers are done
        request_queue.put("STOP")
        gpu_process.join()
        
        print("\n  >> Entrainement du réseau...")

        net.train()
        total_loss = 0
        
        memory_list = list(memory)
        random.shuffle(memory_list)
        
        batches = [memory_list[i:i + batch_size] for i in range(0, len(memory_list), batch_size)]

        for batch in batches:
            states, pis, rewards = zip(*batch)
            
            max_h = max(s.shape[2] for s in states)
            max_w = max(s.shape[3] for s in states)
            padded_states = []
            
            for s in states:
                s_t = torch.from_numpy(s)
                pad_h = max_h - s.shape[2]
                pad_w = max_w - s.shape[3]
                padded_states.append(F.pad(s_t, (0, pad_w, 0, pad_h)))
                
            batch_state = torch.cat(padded_states, dim=0).to(device)
            batch_pi = torch.tensor(np.array(pis), dtype=torch.float32).to(device)
            batch_reward = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)

            optimizer.zero_grad()
            pred_policy, pred_value = net(batch_state)
            
            value_loss = criterion_value(pred_value, batch_reward)
            policy_loss = -torch.sum(batch_pi * log_softmax(pred_policy)) / len(batch)
            loss = value_loss + policy_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  Loss: {total_loss/len(batches) if batches else 0:.4f}")

        # Sauvegarde périodique
        os.makedirs("models", exist_ok=True)
        if (epoch + 1) % 2 == 0:
            checkpoint_path = os.path.join("models", f"alphazero_snake_ep{epoch+1}.pth")
            torch.save(net.state_dict(), checkpoint_path)
            print(f"  [Checkpoint] Modèle sauvegardé avec succès : {checkpoint_path}")

    # Sauvegarde du modèle final
    save_path = os.path.join("models", "alphazero_snake.pth")
    torch.save(net.state_dict(), save_path)
    print(f"\nModele final sauvegarde avec succes dans : {save_path}")

if __name__ == "__main__":
    train_alphazero()
