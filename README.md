# DEEP LEARNING - ALPHAZERO ON A SNAKE GAME 1V1 WITH GRAVITY

**AMIGON Romain et Bardin Clément**

Mai 2026

AlphaZero & PPO appliqués au Snake

Table des matières

1 Le Jeu : Snake 1v1 avec Gravité | 3
1.1 Présentation générale | 3
1.2 Moteur de jeu en C++ | 3
1.2.1 Architecture du moteur | 3
1.2.2 Structures de données principales | 3
1.2.3 Boucle de jeu : la fonction step() | 4
1.2.4 Génération procédurale des niveaux | 4
1.2.5 Interface Python via Pybind11 | 5
1.3 Représentation de l'état pour les agents | 5

2 AlphaZero | 5
2.1 Principe général | 6
2.2 Réseau de neurones : AlphaZeroNet | 6
2.3 Monte Carlo Tree Search (MCTS) | 6
2.3.1 Exploration avec bruit de Dirichlet | 6
2.3.2 Gestion du jeu multi-bot | 7
2.3.3 Limite de profondeur | 7
2.4 Entraînement multiprocessus | 7
2.5 Hyperparamètres et boucle d'entraînement | 7

3 Proximal Policy Optimization (PPO) | 8
3.1 Principe général | 8
3.2 Environnement Gymnasium : SnakePPOEnv | 8
3.2.1 Espace d'observation | 8
3.2.2 Espace d'action | 8
3.2.3 Adversaire intégré | 9
3.2.4 Fonction de récompense | 9
3.3 Extracteur de features : Custom Multi Extractor | 9
3.4 Pipeline d'entraînement | 10
3.4.1 Parallélisme | 10
3.4.2 Hyperparamètres | 10
3.4.3 Reprise automatique de l'entraînement | 10
3.4.4 Callbacks | 10
3.5 Comparaison AlphaZero vs PPO | 11

AlphaZero & PPO appliqués au Snake

## 1 Le Jeu : Snake 1v1 avec Gravité
### 1.1 Présentation générale
Le projet repose sur une variante originale du jeu Snake classique, transformée en un affrontement 1 contre 1 enrichi d'une mécanique de gravité. L'objectif est de concevoir des agents capables d'apprendre à jouer efficacement dans cet environnement complexe, combinant à la fois des dynamiques de survie, de compétition et de physique simulée.
Contrairement au Snake standard, ce jeu introduit plusieurs dimensions stratégiques supplémentaires :
* La gravité force les serpents à rester supportés par des murs, des pommes ou d'autres corps de serpents ; tout segment non supporté tombe d'une case vers le bas à chaque tour.
* Les cartes sont générées procéduralement, assurant une variété quasi-infinie des configurations.
* Chaque équipe peut contrôler plusieurs bots simultanément.
* Un bot est éliminé dès que son corps passe sous 3 segments.
Cette combinaison crée des dynamiques proches du jeu de Tetris, où les déplacements doivent anticiper les chutes futures.

### 1.2 Moteur de jeu en C++
Le moteur du jeu a été entièrement développé en C++ pour des raisons de performance. En effet, lors de l'entraînement par renforcement - en particulier avec MCTS - des millions de simulations de jeu sont effectuées. Un moteur Python pur serait trop lent pour supporter cette charge.

1.2.1 Architecture du moteur
Le code source est organisé dans le répertoire `game/` :
| Fichier | Rôle |
|---|---|
| `include/Coord.hpp` | Structure de coordonnée (x, y) |
| `include/Grid.hpp` | Grille de jeu, tuiles, génération procédurale |
| `include/Game.hpp` | État du jeu, bots, fonction `step()` |
| `include/Utils.hpp` | Utilitaires (directions, vecteurs) |
| `src/Game.cpp` | Logique principale du jeu |
| `src/Grid.cpp` | Génération de niveaux (`GridMaker`) |
| `src/bindings.cpp` | Interface Python via Pybind11 |
Table 1 : Structure du moteur C++

1.2.2 Structures de données principales
L'état du jeu est encapsulé dans la structure `GameState` :
```cpp
struct Bot {
    int id;
    std::deque<Coord> body; // body[0] est toujours la tete
    std::string last_move;
    int tail_reserve; // pommes mangees, croissance en attente
};

struct GameState {
    int width, height;
    Grid grid;
    std::map<int, Bot> bots1; // equipe 1
    std::map<int, Bot> bots2; // equipe 2

    GameState copy() const;
    void step(std::map<int, std::string> my_actions,
              std::map<int, std::string> opp_actions);
};
```
Listing 1 : Extrait de Game.hpp

La grille (`Grid`) est stockée comme une `std::map<Coord, Tile>` et contient également l'ensemble des pommes (`apples`) et des points de spawn.

1.2.3 Boucle de jeu : la fonction step()
La fonction `step()` avance l'état du jeu d'un tour. Elle suit le pipeline suivant :
1.  **Déplacement :** chaque bot se déplace selon son action fournie (ou conserve son dernier mouvement si aucune action n'est spécifiée).
2.  **Détection des collisions :** collisions avec les murs et avec les corps de serpents.
3.  **Consommation de pommes :** si la tête du bot est sur une pomme, `tail_reserve` est incrémenté (la queue pousse aux tours suivants).
4.  **Élimination :** tout bot dont le corps est inférieur à 3 segments est supprimé de l'état.
5.  **Gravité :** itération jusqu'à stabilité ; tout segment de bot non supporté par un mur, une pomme ou un autre bot tombe d'une case vers le bas.

La gravité est le mécanisme le plus complexe. Elle est implémentée en boucle jusqu'à convergence (point fixe) pour gérer les chutes en cascade.

1.2.4 Génération procédurale des niveaux
La classe `GridMaker` génère des cartes de manière procédurale à chaque partie :
*   La taille de la grille varie : hauteur entre 10 et 24 tuiles, largeur ≈ 1.8x la hauteur.
*   Des murs sont placés aléatoirement depuis le bas, avec une densité proportionnelle à la hauteur.
*   Symétrie horizontale : la moitié droite est le miroir de la gauche pour garantir l'équité entre les deux équipes.
*   Les petites poches d'air (zones vides < 10 cases) sont comblées pour éviter les situations bloquées.
*   Des pommes sont placées (2,5% des cases vides minimum) avec un fallback de 4 pommes par côté.
*   Des points de spawn sont générés en tenant compte d'un espacement minimum.

1.2.5 Interface Python via Pybind11
Le moteur C++ est exposé à Python grâce à Pybind11 (`src/bindings.cpp`). Le module compilé `snake_engine.so` est importé directement depuis les scripts Python :
```python
import snake_engine as se

state = se.GameState() # cree un etat avec carte aleatoire
state.step({0: "UP"}, {0: "LEFT"}) # avance d'un tour
print(state.bots1[0].body[0]) # tete du bot 0 equipe 1
```
Listing 2 : Utilisation du moteur depuis Python

La compilation se fait via CMake :
```bash
cd build 
cmake .. 
```

1.3 Représentation de l'état pour les agents
L'état du jeu est encodé en tenseurs pour être consommé par les réseaux de neurones. Deux encodages coexistent selon l'approche :
| Canal | AlphaZero (4 canaux) | PPO (5 canaux) |
|---|---|---|
| 0 | Murs | Murs |
| 1 | Pommes | Corps du joueur (leader) |
| 2 | Corps équipe 1 (avec gradient de tête) | Ennemis |
| 3 | Corps équipe 2 | Pommes |
| 4 | | Alliés |

Pour AlphaZero, l'intensité du gradient sur le canal équipe 1 permet au réseau de distinguer la tête du reste du corps. Pour PPO, un vecteur supplémentaire de 6 valeurs est concaténé : direction normalisée vers la pomme la plus proche et 4 flags d'obstacle adjacent.

## 2 AlphaZero

Voici la version intégralement corrigée de la partie de ton rapport. J'ai expurgé toutes les "hallucinations" de l'IA précédente, réintégré les vraies formules mathématiques et l'architecture exacte de ton code, et ajouté la section sur les résultats décevants mais attendus.

Tu peux copier-coller ce texte directement !

---

### 2. AlphaZero appliqué au Snake

#### 2.1 Principe général
AlphaZero est un algorithme d'apprentissage par renforcement qui apprend à jouer à des jeux de stratégie en partant de zéro, uniquement par auto-jeu (*self-play*). Il combine deux composantes intimement liées :
*   **Un réseau de neurones** $f_\theta(s) \rightarrow (p, v)$ à deux têtes, qui prédit simultanément une politique $p$ (distribution de probabilité sur les actions recommandées) et une valeur $v$ (estimation des chances de victoire depuis l'état actuel).
*   **Un algorithme de Monte Carlo Tree Search (MCTS)** guidé par ce réseau pour explorer l'arbre des possibles et sélectionner les meilleures actions de manière asymétrique.

L'entraînement suit une boucle itérative : l'IA joue contre elle-même pour générer des données de parties (états, probabilités cibles, récompenses finales), puis le réseau est mis à jour sur ces données. Cette nouvelle version du réseau est ensuite utilisée pour générer les parties de l'itération suivante.

#### 2.2 Réseau de neurones : AlphaZeroNet
Le réseau implémenté est un CNN (Convolutional Neural Network) léger, conçu pour analyser efficacement les petites grilles de jeu et permettre des inférences très rapides. Il compte environ 806 885 paramètres.

L'architecture prend en entrée un tenseur représentant l'état du jeu (B, 4, H, W) et se divise comme suit :
*   **Tronc commun (Extraction des caractéristiques) :**
    *   Couche convolutive (32 filtres 3x3, padding 1) + ReLU.
    *   Couche convolutive (64 filtres 3x3, padding 1) + ReLU.
    *   *AdaptiveAvgPool2d(8, 8)* pour garantir une taille de représentation fixe quelle que soit la taille de la grille générée, suivi d'un aplatissement (*Flatten*) donnant un vecteur de 4096 composantes.
*   **Tête de Politique (L'Acteur) :**
    *   Perceptron multicouche (MLP) transformant le vecteur de 4096 vers une couche cachée de 128 neurones (ReLU), puis projetant vers 4 logits correspondant aux directions possibles (Haut, Bas, Gauche, Droite).
*   **Tête de Valeur (Le Critique) :**
    *   MLP projetant le vecteur de 4096 vers une couche cachée de 64 neurones (ReLU), puis vers 1 neurone de sortie. L'activation *Tanh* contraint cette sortie dans l'intervalle $[-1.0, 1.0]$.

#### 2.3 Monte Carlo Tree Search (MCTS) et Exploration
Contrairement aux approches asymétriques où l'adversaire joue au hasard, la génération des parties s'effectue ici en pur *Self-Play* symétrique : le Joueur 1 et le Joueur 2 utilisent tous les deux le MCTS couplé au réseau de neurones pour choisir leurs actions.

*   **Profondeur de recherche :** L'algorithme effectue 25 simulations MCTS par prise de décision pour explorer les branches les plus prometteuses guidées par la formule PUCT.
*   **Bruit de Dirichlet :** Afin d'éviter que l'IA ne joue de manière purement déterministe et s'enferme dans des optimums locaux, du bruit de Dirichlet est injecté artificiellement dans les probabilités calculées à la racine de l'arbre MCTS de chaque tour. Cela force l'agent à explorer des trajectoires imprévues lors de l'entraînement.

#### 2.4 Calcul de la Récompense et Hyperparamètres
La boucle d'entraînement s'exécute séquentiellement. Le signal de récompense $r$, rétropropagé à l'issue de chaque partie (avec un facteur d'atténuation $\gamma = 0.95$), est défini par des conditions strictes :
*   **Victoire par élimination :** $1.0$
*   **Défaite par élimination :** $-1.0$
*   **Fin au temps ou épuisement des ressources :** Comparaison de l'occupation spatiale (longueur totale) des deux camps, selon la formule suivante bridée dans l'intervalle $[-1.0, 1.0]$ :
    $$r = \max\left(-1.0, \min\left(1.0, 1.5 \times \frac{score1 - score2}{score1 + score2}\right)\right)$$

Les paramètres de l'entraînement sont résumés dans le tableau suivant :

| Paramètre | Valeur |
| :--- | :--- |
| Époques | 200 |
| Parties par époque | 100 |
| Simulations MCTS par coup | 25 |
| Taille de batch | 64 |
| Optimiseur | Adam (lr = 0.0005) |
| Perte Policy | Entropie croisée (LogSoftmax) |
| Perte Valeur | Erreur Quadratique Moyenne (MSE) |
| Facteur d'atténuation ($\gamma$) | 0.95 |

#### 2.5 Résultats et limites de l'implémentation
Malgré la pertinence théorique de l'algorithme, les résultats obtenus à l'issue de l'entraînement se sont révélés particulièrement décevants. L'agent peine à développer des stratégies complexes de blocage (spécifiques à Snake) ou à évaluer la valeur réelle d'une position à long terme.

Ces limites étaient cependant attendues au regard de la complexité du problème et des contraintes matérielles :
1.  **Déficit massif de données et de calculs :** AlphaZero tire sa puissance de la loi des grands nombres. Notre volume d'entraînement (20 000 parties, limitées à 25 simulations MCTS par coup) est infinitésimal comparé aux standards de DeepMind (des millions de parties avec au moins 800 simulations par coup). Le réseau manque cruellement d'expérience pour généraliser.
2.  **Architecture CNN trop légère :** Le réseau ne compte que deux couches convolutives et environ 800 000 paramètres. Pour appréhender des concepts spatiaux profonds (comme se représenter un "cul-de-sac" qui va se refermer dans 10 tours), une architecture nettement plus profonde (de type ResNet avec blocs résiduels) serait requise.
3.  **La nature du jeu Snake :** Snake est un environnement à récompenses très retardées. Une action anodine (se diriger vers une pomme) peut entraîner la mort de l'agent 15 tours plus tard. Sans une capacité de prédiction profonde, le signal d'apprentissage généré par notre algorithme reste trop bruité pour permettre l'émergence d'une intelligence tactique supérieure.

---

## 3 Proximal Policy Optimization (PPO)

3.1 Principe général
PPO (Proximal Policy Optimization) est un algorithme de reinforcement learning on-policy développé par OpenAI. À la différence d'AlphaZero, il ne planifie pas à l'avance via une recherche arborescente, mais apprend directement une politique par gradient, en contraignant les mises à jour pour éviter des changements trop brusques.
La fonction objectif PPO est :
$$L^{CLIP}(\theta)=\mathbb{E}_{t}[min(r_{t}(\theta)\hat{A}_{t},clip(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t})]$$
où $r_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{\theta}_{olt}}(a_{t}|s_{t})}$ est le ratio de probabilité et $\hat{A}_{t}$ l'avantage estimé (GAE).
L'implémentation utilise la bibliothèque Stable-Baselines3 (`agent/train_ppo.py`).

3.2 Environnement Gymnasium : SnakePPOEnv
L'environnement PPO est défini dans `agent/ppo_env.py`. Il encapsule le moteur C++ dans une interface standard Gymnasium.

3.2.1 Espace d'observation
L'observation est un dictionnaire à deux composantes :
```python
observation_space = spaces.Dict({
    "image": spaces.Box(0, 255, (5, 50, 50), dtype=np.uint8),
    "vector": spaces.Box(-1.0, 1.0, (6,), dtype=np.float32),
})
```
Listing 4 : Espace d'observation de SnakePPOEnv

*   Image $(5\times50\times50)$ - 5 canaux uint8 : murs, corps du leader, ennemis, pommes, alliés. La grille réelle est copiée dans une fenêtre $50\times50$ (padding de zéros si nécessaire).
*   Vecteur (6) : direction normalisée vers la pomme la plus proche $(\Delta x,\Delta y)$ + 4 flags d'obstacle adjacent (haut, bas, gauche, droite).

3.2.2 Espace d'action
```python
action_space = spaces.Discrete(4) # UP, DOWN, LEFT, RIGHT
```
L'agent contrôle uniquement le bot leader de l'équipe 1. Les autres bots alliés utilisent un BFS vers la pomme la plus proche.

3.2.3 Adversaire intégré
L'équipe adverse est contrôlée par un agent BFS qui navigue vers la pomme la plus proche. Ce choix garantit un adversaire crédible et déterministe sans complexité excessive.

3.2.4 Fonction de récompense
La récompense est dense et multi-composante, conçue pour guider l'apprentissage :
| Événement | Récompense |
|---|---|
| Survie (par pas) | +0.01 |
| Pomme mangée | +1.0 |
| Rapprochement vers une pomme | +0.1 |
| Éloignement d'une pomme | -0.1 |
| Élimination du bot contrôlé | -5.0 |
| Victoire (tous ennemis éliminés) | +10.0 |
| Défaite | -10.0 |
| Boucle détectée | -0.5 |
| Immobilité prolongée | -1.0 |
Table 4 : Fonction de récompense de SnakePPOEnv

La pénalité de boucle est calculée en détectant des positions de tête répétées sur une fenêtre récente, découragent les comportements circulaires stériles.

3.3 Extracteur de features : CustomMultiExtractor
PPO doit traiter un espace d'observation hétérogène (image + vecteur). Un extracteur de features personnalisé est utilisé dans `agent/train_ppo.py` :
```python
class CustomMultiExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        # CNN pour l'image
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # MLP pour le vecteur
        self.vector_net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
        )
        # Fusion
        self.combined = nn.Linear(cnn_out + 64, 512)
```
Listing 5 : Architecture de l'extracteur de features

Les deux branches sont concaténées et projetées dans un espace de 512 dimensions, qui alimente ensuite les têtes politique et valeur de SB3.

3.4 Pipeline d'entraînement
3.4.1 Parallélisme
16 environnements parallèles sont créés via `SubprocVecEnv` pour accélérer la collecte d'expériences.

3.4.2 Hyperparamètres
| Paramètre | Valeur |
|---|---|
| Environnements parallèles | 16 |
| Steps par rollout (`n_steps`) | 1024 |
| Taille de batch | 1024 |
| Époques par mise à jour (`n_epochs`) | 10 |
| Facteur de discount y | 0.99 |
| Lambda GAE | 0.95 |
| Taux d'apprentissage initial | $2.5\times10^{-4}$ (décroissance linéaire) |
| Clip PPO | 0.2 |
Table 5 : Hyperparamètres PPO

3.4.3 Reprise automatique de l'entraînement
Le script détecte automatiquement le checkpoint le plus récent dans `agent/models_v2/` et reprend l'entraînement à partir de celui-ci, permettant des sessions d'entraînement incrémentales sans perte de progression.
```python
checkpoints = sorted(glob("models_v2/ppo_snake_*.zip"),
                     key=lambda f: int(re.search(r'(\d+)_steps', f).group(1)))
if checkpoints:
    model = PPO.load(checkpoints[-1], env=env, ...)
```
Listing 6 : Reprise automatique depuis le dernier checkpoint

3.4.4 Callbacks
Deux callbacks SB3 sont utilisés :
*   `CheckpointCallback` : sauvegarde un checkpoint toutes les 50000 étapes dans `agent/models_v2/`.
*   `EvalCallback` : évalue le modèle périodiquement et conserve le meilleur modèle dans `agent/models_v2/best_model.zip`.
Les métriques d'entraînement sont journalisées dans Tensor Board sous `agent/ppo_snake_tensorboard`.

3.5 Comparaison AlphaZero vs PPO
| Critère | AlphaZero | PPO |
|---|---|---|
| Paradigme | Model-based (MCTS) | Model-free (on-policy) |
| Planification | Oui (50 simulations/coup) | Non |
| Données d'entraînement | Self-play | Interactions env. parallèles |
| Parallélisme | 50 workers + 1 GPU | 16 envs `SubprocVecEnv` |
| Observation | 4 canaux + tenseur | 5 canaux + vecteur 6D |
| Adversaire pendant l'entraînement | Lui-même (self-play) | BFS déterministe |
| Bibliothèque | PyTorch pur | Stable-Baselines3 |
| Vitesse d'inférence | Lente (MCTS) | Rapide (réseau seul) |
Table 6 : Comparaison des deux approches



## Conclusion

Ce projet explore deux paradigmes complémentaires de l'apprentissage par renforcement appliqués à un environnement de jeu original et complexe. AlphaZero offre une planification explicite puissante au prix d'un coût computationnel élevé, tandis que PPO permet un entraînement plus rapide et scalable grâce à la parallélisation d'environnements. 

À l'issue de nos expérimentations, c'est l'algorithme PPO qui a délivré nos meilleurs résultats. Néanmoins, les comportements de nos agents restent basiques et les performances globales sont en deçà d'une véritable maîtrise tactique du jeu. Ce constat, bien que frustrant, était en réalité attendu au regard de la grande complexité de l'environnement (mécanique de gravité, récompenses fortement différées, et dynamique multi-agents).

Ces difficultés ont d'ailleurs été largement corroborées par la communauté lors de la compétition CodinGame associée à ce jeu. Le forum mis en place pour le partage de stratégies (https://www.codingame.com/forum/t/winter-challenge-2026-feedbacks-and-strategies/208044/2) a mis en évidence que le haut du classement était dominé quasi exclusivement par des méthodes classiques : les meilleures approches reposaient sur des heuristiques poussées, des algorithmes de recherche de chemin (A*), des arbres de recherche symboliques (Minimax, Smitsimax), ... . 

Bien que les méthodes de Deep Learning (PPO, AlphaZero) aient suscité un très fort engouement sur ce même forum, elles ont surtout généré de longues discussions autour des difficultés d'apprentissage, d'instabilité des modèles et de nombreuses demandes de conseils. Lors du partage des stratégies finales en fin de compétition, un seul et unique participant a rapporté avoir réussi à faire fonctionner PPO de manière compétitive. 

Cette rareté des succès vient confirmer et valider nos propres résultats : appliquer efficacement le Reinforcement Learning sur un tel jeu requiert une ingénierie des récompenses et des architectures bien au-delà des standards habituels. Quoi qu'il en soit, le développement d'un moteur C++ performant exposé via Pybind11, couplé à l'intégration de ces deux algorithmes d'état de l'art, constitue une base technique solide et une expérience d'apprentissage inestimable sur les limites et les défis de l'IA moderne.