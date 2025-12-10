import numpy as np

class MyAgent():
    def __init__(self, num_agents: int):        
        self.rng = np.random.default_rng()
        self.num_agents = num_agents
        self.previous_positions = {}  # Mémoire locale pour éviter les boucles
        self.stuck_counter = {}  # Détecter les blocages
    
    def get_action(self, state: list, evaluation: bool = False):
        actions = []
        for i in range(self.num_agents):
            agent_state = state[i]
            position, orientation, goal, lidars, nearby_agents = self.parse_state(agent_state)
            
            # Calcul de la direction optimale
            optimal_direction = self.compute_optimal_direction(position, goal)
            
            # Détermination de l'action
            action = self.decide_action(position, orientation, optimal_direction, lidars, nearby_agents, i)
            
            print(f"Agent {i}: Position {position}, Orientation {orientation}, Goal {goal}, Optimal Direction {optimal_direction}, Action {action}")
            
            actions.append(action)
        return actions
    
    def parse_state(self, agent_state):
        """ Extrait les informations de l'état de l'agent """
        position = tuple(agent_state[:2])
        orientation = agent_state[2]
        goal = tuple(agent_state[3:5])
        lidars = {
            'front': agent_state[5:7], 
            'right': agent_state[7:9], 
            'left': agent_state[9:11]
        }
        nearby_agents = agent_state[11:]
        return position, orientation, goal, lidars, nearby_agents

    def compute_optimal_direction(self, position, goal):
        """ Calcule la meilleure direction pour se rapprocher du but """
        dx, dy = goal[0] - position[0], goal[1] - position[1]
        if abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'up' if dy > 0 else 'down'

    def decide_action(self, position, orientation, optimal_direction, lidars, nearby_agents, agent_id):
        """ Détermine la meilleure action en fonction des capteurs et de la direction optimale """
        if position in self.previous_positions:
            self.stuck_counter[agent_id] = self.stuck_counter.get(agent_id, 0) + 1
            if self.stuck_counter[agent_id] > 3:  # Si bloqué pendant plusieurs tours
                safe_moves = []
                if lidars['front'][0] >= 2:
                    safe_moves.append(1)
                if lidars['left'][0] >= 2:
                    safe_moves.append(3)
                if lidars['right'][0] >= 2:
                    safe_moves.append(4)
                if safe_moves:
                    return self.rng.choice(safe_moves)  # Avancer vers une case safe
                return self.rng.choice([5, 6])  # Tourner aléatoirement
        else:
            self.stuck_counter[agent_id] = 0
        
        # Éviter de revenir en arrière
        if self.previous_positions.get(agent_id) == position:
            return self.rng.choice([5, 6])  # Tourner pour explorer
        
        # Vérifier le nombre d'obstacles à proximité
        obstacles = sum(1 for d in lidars.values() if d[0] < 2)
        if obstacles == 3:
            return self.rng.choice([5, 6])  # Pivoter si bloqué
        elif obstacles == 2:
            safe_moves = [k for k, v in zip(['up', 'down', 'right', 'left'], [1, 2, 4, 3]) if lidars[k][0] >= 2]
            if safe_moves:
                return self.rng.choice([action_mapping[safe_moves[0]]])  # Avancer vers une case safe
            return self.rng.choice([5, 6])  # Tourner pour analyser
        
        # Suivre la direction optimale
        action_mapping = {
            'up': 1, 'down': 2, 'right': 4, 'left': 3
        }
        return action_mapping.get(optimal_direction, 0)
    
    def update_policy(self, actions: list, state: list, reward: float):
        """ Pas d'apprentissage, juste mise à jour de la mémoire """
        for i, agent_state in enumerate(state):
            self.previous_positions[i] = tuple(agent_state[:2])