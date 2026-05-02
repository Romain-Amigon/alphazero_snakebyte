import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
from ppo_env import SnakePPOEnv
from typing import Callable

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Décroissance du learning rate (linéaire).
    :param initial_value: Le learning rate initial.
    :return: Une fonction qui prend en entrée la progression (de 1 à 0) et calcule le LR actuel.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining va de 1 (début de l'entraînement) à 0 (fin)
        return progress_remaining * initial_value
    return func

class CustomMultiExtractor(BaseFeaturesExtractor):
    """
    Extracteur personnalisé pour traiter l'observation Dictionnaire (Image + Vecteur).
    L'extracteur va traiter l'image avec un CNN, et concaténer le résultat avec le vecteur.
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        img_space = observation_space.spaces['image']
        n_input_channels = img_space.shape[0]
        
        # stride=2 sur les 2 premières couches : (5,50,50) → (32,25,25) → (64,13,13) → (64,13,13)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Calcul de la taille de sortie du Flatten()
        with torch.no_grad():
            sample_img = torch.as_tensor(img_space.sample()[None]).float()
            n_flatten = self.cnn(sample_img).shape[1]
            
        vec_space = observation_space.spaces['vector']
        vec_dim = vec_space.shape[0]
            
        # Couche combinée (Image + Vecteur)
        self.linear = nn.Sequential(
            nn.Linear(n_flatten + vec_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: dict) -> torch.Tensor:
        img_features = self.cnn(observations["image"])
        vec_features = observations["vector"]
        combined = torch.cat((img_features, vec_features), dim=1)
        return self.linear(combined)

def make_env():
    """
    Fonction utilitaire pour créer une instance de l'environnement.
    Ceci est nécessaire pour la parallélisation (SubprocVecEnv).
    """
    def _init():
        # L'envelopper dans un Monitor permet à StableBaselines de traquer 
        # la récompense totale (ep_rew_mean) et la longueur de l'épisode (ep_len_mean)
        env = SnakePPOEnv(max_steps=500)
        env = Monitor(env)
        return env
    return _init

def train_ppo(total_timesteps=20_000_000, resume_model_path=None):
    print("Initialisation de l'environnement Gym pour PPO avec Multiprocessing...")
    

    num_envs = 16
    
    # SubprocVecEnv lance un processus séparé (multiprocessing) pour chaque environnement.
    vec_env = SubprocVecEnv([make_env() for i in range(num_envs)])

    # Custom CNN is required because NatureCNN expects larger images
    policy_kwargs = dict(
        features_extractor_class=CustomMultiExtractor,
        features_extractor_kwargs=dict(features_dim=256),
    )

    if resume_model_path and os.path.exists(resume_model_path):
        print(f"Reprise de l'entraînement à partir du modèle: {resume_model_path}")

        # Hyperparams de reprise plus agressifs pour sortir d'un minimum local :
        #  - LR assez haut pour corriger un collapse (mais pas autant qu'au début)
        #  - ent_coef relevé pour forcer l'exploration
        #  - clip_range standard pour permettre de vraies mises à jour
        #  - n_epochs réduit pour limiter l'overfitting aux rollouts récents
        # Reprise après fix reward : LR modéré, batch & epochs plus importants
        # pour mieux exploiter le GPU. ent_coef relevé pour ré-explorer.
        custom_objs = {
            "policy_kwargs": policy_kwargs,
            "learning_rate": 3e-5,
            "ent_coef": 0.02,
            "clip_range": 0.2,
            "n_epochs": 10,
            "batch_size": 1024,
            "n_steps": 1024,
        }

        model = PPO.load(
            resume_model_path,
            env=vec_env,
            custom_objects=custom_objs,
            device="cuda",
            tensorboard_log="./ppo_snake_tensorboard_v2/"
        )
    else:
        # n_steps=1024 × num_envs=16 = 16384 transitions/rollout
        # batch_size=1024 → 16 mini-batches → bonne utilisation GPU
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            verbose=1,
            learning_rate=linear_schedule(2.5e-4),
            n_steps=1024,
            batch_size=1024,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            max_grad_norm=0.5,
            ent_coef=0.015,
            policy_kwargs=policy_kwargs,
            device="cuda",
            tensorboard_log="./ppo_snake_tensorboard_v2/"
        )

    # Sauvegarde périodique (rollback)
    os.makedirs("models_v2", exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path='./models_v2/',
        name_prefix='ppo_snake_v2'
    )

    # Eval séparé → sauvegarde automatique du MEILLEUR modèle selon ep_rew_mean
    eval_env = DummyVecEnv([make_env()])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models_v2/best/',
        log_path='./models_v2/eval_logs/',
        eval_freq=25000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    callbacks = [checkpoint_callback, eval_callback]

    print(f"Lancement de l'entraînement pour {total_timesteps} pas...")
    reset_timesteps = resume_model_path is None
    
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks, reset_num_timesteps=reset_timesteps)
    except KeyboardInterrupt:
        print("\nEntraînement interrompu manuellement. Sauvegarde de sécurité en cours...")
        model.save(os.path.join("models_v2", "ppo_snake_v2_interrupted.zip"))
        print("Modèle de secours sauvegardé sous models_v2/ppo_snake_v2_interrupted.zip")
        sys.exit(0)

    # Sauvegarde finale
    save_path = os.path.join("models_v2", "ppo_snake_v2_final.zip")
    model.save(save_path)
    print(f"Entraînement terminé. Modèle final sauvegardé sous {save_path}")

if __name__ == "__main__":
    import sys
    import glob
    
    # Trouver automatiquement le modèle .zip le plus récent si aucun n'est passé en argument
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        list_of_files = glob.glob('models_v2/ppo_snake*.zip')
        if list_of_files:
            path = max(list_of_files, key=os.path.getmtime)
            print(f"Modèle le plus récent trouvé automatiquement : {path}")
        else:
            path = None
            
    if path:
        train_ppo(resume_model_path=path)
    else:
        train_ppo()
