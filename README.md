# DEEP LEARNING - ALPHAZERO ON A SNAKE GAME 1V1 WITH GRAVITY

**AMIGON Romain** <span style="font-size: 0.5em;">et Bardin Clément</span>


### Moteur de Simulation Hybride C++ / Python (AlphaZero)
Ce projet implémente un environnement de simulation ultra-rapide optimisé pour l'entraînement d'algorithmes d'intelligence artificielle par renforcement (de type AlphaZero / MCTS).

Afin de garantir des performances maximales lors des millions de simulations requises par l'auto-apprentissage, le cœur logique et physique du jeu a été entièrement développé en C++. Cet engin natif est ensuite compilé et exposé sous forme de module natif à Python, offrant ainsi le meilleur des deux mondes : la vitesse d'exécution brute du C++ et la flexibilité de Python pour le développement de modèles Deep Learning.

### Architecture et Flux d'Exécution
Le système repose sur une boucle d'interaction stricte entre le moteur compilé et le script de contrôle Python.

Initialisation : Le script principal main.py instancie l'objet GameState (C++). La grille, les murs, les énergies et les entités sont générés en mémoire native.

Observation : L'état actuel (positions, cartes) est lu par Python et transmis à l'agent décisionnel (agent.py).

Décision : L'agent évalue les mouvements possibles (via des heuristiques ou un réseau de neurones) et retourne un dictionnaire d'actions pour chaque entité.

Résolution (Step) : Les actions sont injectées dans la fonction step() du moteur C++. Le C++ calcule instantanément la physique du tour : déplacements, collisions, ramassage d'énergie, gravité et effondrements.

Rendu : L'état mis à jour est lu par Pygame pour l'affichage graphique, et le cycle recommence.

## Résumé des Classes Core (C++)
Le moteur repose sur une architecture orientée objet minimaliste et optimisée :

GameState :Contient la grille et les dictionnaires de joueurs (bots1, bots2). Exécute la fonction critique step() qui résout un tour complet de simulation.

Grid : Représente la carte spatiale du jeu (aplatie en 1D pour la performance). Gère les accès ultra-rapides aux cases, la position des pommes (apples) et les points d'apparition (spawns).

Bot : Représente une entité physique (serpent/robot). Gère l'historique de ses positions corporelles (via une liste de Coord) et son statut de survie.

Tile & Coord : Structures de données fondamentales pour la typologie des cases (TYPE_WALL, TYPE_EMPTY) et la manipulation mathématique des coordonnées 2D.

### Compilation et Intégration Python
Le pont entre le C++ et le Python est assuré par Pybind11 et CMake.
La compilation génère un fichier dynamique natif (.pyd sous Windows, .so sous Linux/Mac) qui se comporte exactement comme une bibliothèque Python standard.

Les structures de données complexes du C++ (comme les std::vector, std::set et std::map) sont automatiquement et silencieusement traduites en listes, sets et dictionnaires Python lors des appels, garantissant une intégration transparente pour la création de l'IA.