#encoding:UTF-8
import argparse
import heapq
import multiprocessing
from matplotlib import patches
import numpy as np
import random
import copy
import seaborn
import yaml
from environment import Env
from collections import Counter, defaultdict
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from abc import ABC, abstractmethod
from pathlib import Path
from mluti_paths import MultiPathGenerator

class Agent(ABC):
    """Base agent class"""
    def __init__(self, start_state, goal_state, agent_id):
        self.state = start_state      # Current state
        self.goal = goal_state        # Goal state
        self.back_point = None       # Backtrack point
        self.path = [start_state]     # Path taken
        self.agent_id = agent_id      # Agent unique identifier
        self.num_neighbors = 0        # Number of neighbors
        
    @abstractmethod
    def get_next_state(self, occupied_positions):
        """Get next state (to be implemented in subclass)"""
        pass

class HumanAgent(Agent):
    """Human agent class"""
    def __init__(self,  env, start_state, goal_state, agent_id, factor):
        super().__init__(start_state, goal_state, agent_id)
        self.neighbors_cache = {}  # Neighbors cache
        self.factor = factor # Action factor
        self.env = env

    def get_neighbors(self):
        """Get valid neighbor states (considering obstacles and collisions)"""
        if self.state not in self.neighbors_cache:
            self.neighbors_cache[self.state] = self.env.get_legal_neighbors_human(self.state)
        return self.neighbors_cache[self.state]
    
    def get_next_state(self, occupied_positions):
        """Calculate next state for human agent (with collision detection)"""
        if self.state == self.goal:
            return self.state
            
        # Get candidate neighbors (exclude occupied positions)
        occupied_set = set(occupied_positions)
        next_states = [s for s in self.get_neighbors() if s not in occupied_set]
        
        # Goal priority check
        if self.goal in next_states:
            return self.goal

        # Handle waiting action
        if len(next_states) <= 1:
            return self.state
        else:
            some_condition = [self.state]
        
        # Exclude backtrack point
        if self.back_point and self.back_point != self.state:
            some_condition.append(self.back_point)
        checked_next_states = [s for s in copy.deepcopy(next_states) if s not in some_condition]

        if not checked_next_states:
            # Backtrack mechanism
            return self.back_point
            
        # Dynamic weight calculation
        distances = [self.env.manhattan_distance(s, self.goal) if s not in some_condition else float('inf') for s in next_states]
        min_dist = min(distances)
        min_indices = [i for i, d in enumerate(distances) if d == min_dist]
        n_valid = len(min_indices)
        weights = [0.0]*len(next_states)
        value = 1 - (len(next_states)-n_valid)*self.factor
        base_weight = value / n_valid
        weights = [base_weight if i in min_indices else self.factor for i,_ in enumerate(next_states)]

        # Random exploration for other states
        if len(next_states) != len(weights):
             print(f"Error: next_state length {len(next_states)}, probabilities length {len(weights)}")
        chosen = random.choices(next_states, weights=weights, k=1)[0]

        if chosen != self.state and self.state in occupied_positions:
            occupied_positions.remove(self.state)
        self.back_point = self.state if chosen != self.state else self.back_point
        return chosen

class MultiHumanSimulator:
    """Multi-agent simulator"""
    def __init__(self, env, human_start, human_goal, n_humans, steps_per_sim=30):
        self.env = env
        self.num_agents = n_humans
        self.steps_per_sim = steps_per_sim
        self.agents = [HumanAgent(
            env,
            start_state = human_start[i],
            goal_state = human_goal[i],
            factor = 0.1,
            agent_id=i
        ) for i in range(n_humans)]
        self.conflict_count = 0

    def _get_next_state(self, agent, occupied):
        """Calculate agent's next state (with collision detection)"""
        return agent.get_next_state(occupied)
    
    def _simulate_step(self):
        """Simulate one step for all agents"""
        all_neigibors = []
        for agent in self.agents:
            neighbors = agent.get_neighbors()
            agent.num_neighbors = len(neighbors)
            all_neigibors.extend(neighbors)
        neighbors_count = Counter(all_neigibors)
        occupied = [point for point, count in neighbors_count.items() if count > 1]
        
        # Sort agents by neighbor count to avoid conflicts
        self.agents.sort(key=lambda x: x.num_neighbors, reverse=True)
        
        # Update agent states (conflict resolution: prioritize agents in original positions)
        for agent in self.agents:
            next_state = self._get_next_state(agent, occupied)
            agent.path.append(next_state)
            agent.state = next_state

        # Check for conflicts between agents
        for agent in self.agents:
            conflict = any(
                agent.state == other_agent.state 
                for other_agent in self.agents 
                if agent.agent_id != other_agent.agent_id)
            if conflict:
                self.conflict_count += 1
    
    def simulate(self, num_sims=2000):
        """Execute multiple simulations"""
        all_paths = []
        for _ in range(num_sims):
            # Reset all agent states
            for agent in self.agents:
                agent.state = agent.path[0]
                agent.path = [agent.path[0]]
            
            # Run complete simulation
            for _ in range(self.steps_per_sim):
                self._simulate_step()
            
            # Collect paths
            all_paths.extend([agent.path for agent in self.agents])
        return all_paths
    
    def reset(self):
        """Reset simulator state"""
        for agent in self.agents:
            agent.state = agent.path[0]
            agent.path = [agent.path[0]]

class RiskCalculator:
    """Risk calculator (supports multi-agent)"""
    def __init__(self, all_paths, total_steps, num_sims, show = False):
        self.all_paths = all_paths
        self.total_steps = total_steps+1
        self.num_sims = num_sims
        self.show = show

    def calculate_risk(self,env):
        """Calculate comprehensive risk map"""
        risk_map = defaultdict(lambda: defaultdict(float))
        
        for step in range(self.total_steps):
            # Count all agent positions at this time step
            position_counts = Counter()
            for path in self.all_paths:
                if step < len(path):
                    position_counts[path[step]] += 1
            
            # Calculate risk values
            for pos, count in position_counts.items():
                density = count / self.num_sims
                risk = density
                risk_map[step][pos] = risk
        
        # Visualize risk map
        if self.show: 
            all_occu = copy.deepcopy(risk_map)
            for step in range(self.total_steps):
                fig = plt.figure(figsize=(10,10))
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.tick_top()
                grid = [[0 for _ in range(env.width)] for _ in range(env.height)]
                for v in env.obstacles:
                    grid[v[1]][v[0]] = 1 
                for i in all_occu[step]:
                    grid[i[1]][i[0]] += all_occu[step][i]

                seaborn.heatmap(grid, square= True, cbar= False, cmap="Greys", linewidths = 1, linecolor="black", annot= True, fmt=".3g")
                for i in all_occu[step]:
                    ax.add_patch(patches.Rectangle((i[0], i[1]),1,1,edgecolor = 'red',facecolor = 'red',fill=False))

                ax.set_title('j')
                ax.set_ylabel('i')
                fig.savefig("log_path/constraint_"+"_TH_"+str(len(all_occu))+"_step_"+str(step)+".pdf")
                plt.clf()
                plt.close()
        return risk_map
#----------------------------------------
class AStarAgent:
    def __init__(self, env):
        self.env = env

    def heuristic(self, a, b):
        """Manhattan distance heuristic function"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star_search(self, start, goal):
        """A* search algorithm"""
        open_set = []
        # Add random number to break ties
        heapq.heappush(open_set, (self.heuristic(start, goal), random.random(), start))
        came_from = {}
        g_score = {pos: float('inf') for pos in self.env.able_state}
        g_score[start] = 0
        f_score = {pos: float('inf') for pos in self.env.able_state}
        f_score[start] = self.heuristic(start, goal)

        while open_set:
            # Pop and ignore random number used for sorting
            current_f, _, current = heapq.heappop(open_set)

            if current == goal:
                # Path reconstruction
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Randomize action order
            for action in range(self.env.n_actions):
                next_state, _ = self.env.step(current, action, 'path')
                if next_state not in self.env.able_state:
                    continue
                tentative_g_score = g_score[current] + 1

                if tentative_g_score < g_score[next_state]:
                    came_from[next_state] = current
                    g_score[next_state] = tentative_g_score
                    new_f = tentative_g_score + self.heuristic(next_state, goal)
                    f_score[next_state] = new_f
                    # Add random number to break ties
                    heapq.heappush(open_set, (new_f, random.random(), next_state))
        return None
#----------------------------------------
class QLearningAgent:
    def __init__(self, env, actions, n_episodes):
        self.actions = actions
        self.n_episodes = n_episodes
        self.env = env
        # Pre-cache legal actions for all possible states
        self.legal_actions_cache = defaultdict(list)
        for state in env.able_state:
            self._cache_legal_actions(state)

    def _cache_legal_actions(self, state):
        """Pre-calculate legal actions for specified state"""
        valid_actions = []
        for action in self.actions:
            next_state = self.env.get_state_action_space(state, action)
            if next_state in self.env.able_state:
                valid_actions.append(action)
        self.legal_actions_cache[state] = valid_actions

    def q_learning(self,alpha=0.1,gamma=0.9,epsilon=0.4,start = None, goals = None):
        # Training
        Q = {}
        reward_iteration = []

        for i in tqdm(range(self.n_episodes), leave=False):
            state = start
            episode_reward = 0.0
            while state != goals:
                # Fast Q-table entry access or initialization
                if state not in Q:
                    Q[state] = {a: 0.0 for a in self.legal_actions_cache[state]}
                action = self._select_action(Q[state], epsilon)
                next_state,reward = self.env.step(state,action,'train')
                episode_reward += reward

                # Pre-calculate next_state Q-table entry
                if next_state not in Q:
                    Q[next_state] = {a: 0.0 for a in self.legal_actions_cache[next_state]}

                # Bellman equation update
                max_q = max(Q[next_state].values()) if Q[next_state] else 0
                
                current_q = Q[state][action] 
                Q[state][action] += alpha * (reward + gamma * max_q - current_q)

                state = next_state
                
            reward_iteration.append(episode_reward)
        return Q, reward_iteration
    
    def q_learning_with_human(self,alpha=0.1,gamma=0.9,epsilon=0.4,start = None,goals = None,human_simulator = None):
        Q = defaultdict(lambda:{})

        reward_iteration = []
        for i in tqdm(range(self.n_episodes), leave=False):
            state = start
            episode_reward = 0.0
            human_simulator.reset()
            while state != goals:
                if state not in Q:
                    Q[state] = {a: 0.0 for a in self.legal_actions_cache[state]}
                
                action = self._select_action(Q[state], epsilon)
                next_state,reward = self.env.step(state,action,'train')
                human_state = [human.state for human in human_simulator.agents]
                human_simulator._simulate_step()
                for i,human in enumerate(human_simulator.agents):
                    if (human.state == next_state):
                        reward -= 2
                    if (human.state == state and human_state[i] == next_state):
                        reward -= 2

                    # Calculate Q-value for current state-action
                    episode_reward += reward

                if next_state not in Q:
                    Q[next_state] = {a: 0.0 for a in self.legal_actions_cache[next_state]}

                max_q = max(Q[next_state].values()) if Q[next_state] else 0
                current_q = Q[state][action] 
                Q[state][action] += alpha * (reward+gamma*max_q - current_q)
                
                state = next_state
        
            reward_iteration.append(episode_reward)
        return Q, reward_iteration

    def _select_action(self, action_values, epsilon):
        """Optimization: use numpy for faster random selection"""
        if np.random.random() < epsilon:
            return np.random.choice(list(action_values.keys()))
        max_value = max(action_values.values())
        max_actions = [a for a, v in action_values.items() if v == max_value]
        return np.random.choice(max_actions)
    
    
#----------------------------------------
class Stack:
    def __init__(self):
        self.items = []  # Store stack elements (tuple(s),i,list[actions],state_pos)
        self.entry_map = set() # For fast element existence check

    def is_empty(self):
        return len(self.items) == 0  # Check if stack is empty

    def push(self, item):
        state ,path_idx,_,_ = item
        self.items.append(item)  # Push operation
        self.entry_map.add((state,path_idx))

    def contains(self,state,path_idx):
        # Check if stack contains specified state and path index
        return (state,path_idx) in self.entry_map
    
    def pop_action(self,path_idx):
        for i in reversed(range(len(self.items))):
            item = self.items[i]
            if item[1] == path_idx and item[2]:
                action = item[2].pop()
                if not item[2]:  # Remove if action list is empty
                    self.items.pop(i)
                return action
        return None
    
    def pop(self):
        if self.is_empty():
            raise IndexError("pop from empty stack")
        return self.items.pop() # Last pushed state

    def update_stack(self):
        if self.is_empty():
            raise IndexError("peek from empty stack")
        if len(self.items[-1][2]) >= 1:
            a = self.items[-1][2].pop()
            if not self.items[-1][2]:  # Remove state if action list is empty
                removed_item = self.items.pop()
                self.entry_map.remove((removed_item[0],removed_item[1]))    
            return a  # Get optimal action from top element
        return None
    
    def peek(self):
        if self.is_empty():
            return None
        else:
            return self.items[-1]  # View top element
    def peek_state(self):
        if self.is_empty():
            return None
        else:
            return self.items[-1][0]  # View top element state
    def _items(self):
        return [item[0] for item in self.items]
    
    def size(self):
        return len(self.items)  # Return stack size
 
def limit(all_actions, previous_action=None):
    """
    Limit action sequence to avoid consecutive opposite actions
    """
    if previous_action is None:
        return all_actions  # Return all actions if no previous action
    
    # Define opposite action mapping
    opposite_actions = {
        0 : 1,
        1 : 0,
        2 : 3,
        3 : 2
    }
    
    # Filter out actions opposite to previous action
    return [action for action in all_actions if action != opposite_actions.get(previous_action)]
#----------------------------------------
def path_generator_with_human(env, Q_dict, start, agent_goals, budget):
    state = start
    path = [state]
    t = 0
    while state != agent_goals and t < budget:
        # Get Q-values for all available actions at current state
        action_values = Q_dict.get(state)
        
        # Select actions with maximum Q-value
        max_q = max(action_values.values())
        best_actions = [a for a ,q in action_values.items() if q == max_q]
        
        unvisited_optimal = []
        for action in best_actions:
            next_state, _ = env.step(state, action, 'path')
            if next_state not in path:
                unvisited_optimal.append((action,next_state))

        # 3. Unvisited optimal actions exist: randomly choose one
        if unvisited_optimal:
            chosen_action, next_state = random.choice(unvisited_optimal)
        
        # 4. No unvisited optimal actions: global search by Q-value descending order
        else:
            sorted_actions = sorted(action_values.items(), key=lambda x: -x[1])
            for action, q in sorted_actions:
                next_state, _ = env.step(state, action, 'path')
                if next_state not in path:
                    chosen_action = action
                    break
            else:  # All actions lead to cycles, force choose optimal action
                chosen_action = random.choice(best_actions)
                next_state, _ = env.step(state, chosen_action, 'path')
        state = next_state
        path.append(state)
        t += 1
   
    return path

def SIM(env, path_set, beta=0.5):
    """Filter multiple paths to optimize runtime"""
    
    if not path_set:
        return []
    
    # Step 1: Select initial base paths
    min_length = min(len(path) for path in path_set)
    base_paths = [path for path in path_set if len(path) == min_length]
    
    # Calculate edge point count
    verge_point = []
    for path in base_paths:
        count = 0
        for point in path:
            if 0 in point or ('size' in globals() and size[0] in point):  # Assume size is external variable
                count += 1
        verge_point.append(count)
    
    max_verge = max(verge_point)
    base_paths = [path for path, vp in zip(base_paths, verge_point) if vp == max_verge]
    
    better_path = base_paths.copy()
    path_set = [p for p in path_set if p not in better_path]
    
    if not path_set:
        return better_path
    
    # Step 2: Build initial similarity matrix
    value_array = np.array([[view_distance_with_cosine(env, p, bp)[0] for p in path_set] for bp in better_path])
    
    # Step 3: Dynamic path filtering
    while (len(path_set)>1):
        # Calculate mean and generate filter mask
        mean_value = np.mean(value_array, axis=1, keepdims=True)
        mask = np.all(value_array >= beta * mean_value, axis=0)
        filtered_indices = np.where(mask)[0]
        
        if filtered_indices.size == 0:
            break
            
        # Update path set and similarity matrix
        path_set = [path_set[i] for i in filtered_indices]
        value_array = value_array[:, filtered_indices]
        
        # Select optimal candidate path
        min_values = np.min(value_array, axis=0)
        best_idx = np.argmax(min_values)
        selected_path = path_set[best_idx]
        
        # Update data structure
        better_path.append(selected_path)
        path_set.pop(best_idx)
        value_array = np.delete(value_array, best_idx, axis=1)
        
        # Calculate similarity for new path and merge into matrix
        new_row = np.array([view_distance_with_cosine(env, p, selected_path)[0] for p in path_set])
        value_array = np.vstack([value_array, new_row]) if value_array.size else new_row.reshape(1, -1)

    return better_path

#---------------------------------------
def view_distance_with_cosine(env, path_new,path_contrast):
    """
    Distance and cosine similarity calculation
    path_new: new path
    path_contrast: reference path
    """

    # Path lengths
    l = len(path_new)
    L = len(path_contrast)

    # Fill shorter path with goal position to make both paths same length
    if L < l:
        longpath = path_new
        current_path = path_contrast + [path_contrast[-1]]*(l-L) # Fill short path
    elif L > l:
        longpath = path_contrast
        current_path = path_new + [path_new[-1]]*(L-l)
    else:
        longpath = path_new
        current_path = path_contrast

    # Prepare for similarity calculation
    current_path_new_array = np.array(current_path) - current_path[0]
    current_path_new_array = current_path_new_array[1:]
    path_contrast_array = np.array(path_contrast) - path_contrast[0]
    path_contrast_array = path_contrast_array[1:]
    path_new_array = np.array(path_new) - path_new[0]
    path_new_array = path_new_array[1:]

    # Calculate cosine similarity
    dot_product = np.sum(current_path_new_array*path_contrast_array,axis=1) if L>l else np.sum(path_new_array*current_path_new_array,axis=1)
    norm_path_l = np.linalg.norm(current_path_new_array,axis=1) if L>l else np.linalg.norm(path_new_array,axis=1)
    norm_path_L = np.linalg.norm(path_contrast_array,axis=1) if L>l else np.linalg.norm(current_path_new_array,axis=1)
    multiplity = norm_path_l*norm_path_L
    cosine = dot_product/multiplity
    n_cosine = -cosine + 1 
    
    # Calculate distance similarity
    dis = [2/(env.height+env.width)*distance(x,y) for x,y in zip(longpath[1:],current_path[1:])] 
    
    # Combine similarity measures
    smi_max = [min(x,y) for x,y in zip(n_cosine,dis)]
    
    # Average similarity value
    maxvalue = sum(smi_max)/len(smi_max)

    return [maxvalue]

#----------------------------------------

def distance(point1, point2):
    """
    Calculate distance between two points: Manhattan distance
    """
    d = abs(point1[0]-point2[0]) + abs(point1[1]-point2[1])
    return d

def jaccard_similarity(path1, path2):## 1: same path, 0: completely different
    """Calculate Jaccard similarity between two paths"""
    set1 = set(path1)
    set2 = set(path2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

#----------------------------------------
def safer_path(path_set,risk_map):
    """
    Safe path selection in environment with random human movement
    """
    risk_set = []
    
    # Calculate risk value for each path: r = sum(Pt(i,j))
    for path in path_set: 
        r = 0
        for i in range(len(path)):
            r += risk_map[i].get(path[i]) if path[i] in risk_map[i].keys() else 0
            if i < len(path)-1:
                r += risk_map[i].get(path[i+1]) * risk_map[i+1].get(path[i]) if path[i] in risk_map[i+1].keys() and path[i+1] in risk_map[i].keys() else 0     
        risk_set.append(r)
    min_risk = min(risk_set)
    min_indices = [i for i,risk in enumerate(risk_set) if risk == min_risk]

    random_index = random.choice(min_indices)
    return path_set[random_index]

#----------------------------------------
def simulation(env,step):
	# Human path at each time step
    plan = []
    state = human_start
    plan.append(state)
    back_point = None

    # Simulate human walking path, record all states
    for _ in range(step):
        next_state = env.human_next_state(state,human_goal,back_point)
        plan.append(next_state)
        back_point = state if next_state != state else back_point
        state = next_state

    return plan

#----------------------------------------
def risk_map(env,sim=2000, step=30, recoverd=False):
    """
    Path risk calculation: count records state occurrences at each time step
    """
    human_simulation = []
    risk_map = {}

    for i in range(sim):
        path = simulation(env,step)
        human_simulation.append(path) # Collect all paths
    
    for i in range(step):
        count = Counter([p[i] for p in human_simulation]) # Count state occurrences at time step i
        count = {k:v/sim for k,v in count.items()}
        risk_map.update({i:count})

    # Draw heatmap with probability of each vertex
    if recoverd: 
        all_occu = copy.deepcopy(risk_map)
        for step in range(1,30):
            fig = plt.figure(figsize=(10,10))
            ax = fig.add_subplot(1, 1, 1)
            ax.xaxis.tick_top()
            grid = [[0 for _ in range(env.width)] for _ in range(env.height)]
            for v in env.obstacles:
                grid[v[1]][v[0]] = 1 
            for i in all_occu[step]:
                grid[i[1]][i[0]] += all_occu[step][i]

            seaborn.heatmap(grid, square= True, cbar= False, cmap="Greys", linewidths = 1, linecolor="black", annot= True, fmt=".3g")
            for i in all_occu[step]:
                ax.add_patch(patches.Rectangle((i[0], i[1]),1,1,edgecolor = 'red',facecolor = 'red',fill=False))

            ax.set_title('j')
            ax.set_ylabel('i')
            fig.savefig("log_path/constraint_"+"_TH_"+str(len(all_occu))+"_step_"+str(step)+".pdf")
            plt.clf()
            plt.close()

    return risk_map

def Evaluation_collision_with_human(path,simulator,agent_goals):
    """
    Simulate agent following safest path while humans start from their positions
    :param agent_path: Agent's safe path
    :param human_start: Human starting positions
    :param env: Environment object
    :param steps: Number of simulations
    :return: Collision count
    """
    collision_count = 0
    agent_state = path[0]
    human_path = []
    
    length = len(path)
    collision_position = []
    collision_step = []
    conflict = False
    reward = length*(-0.1)
    if path[-1] in agent_goals:
        reward += 10
    simulator.reset()
    
    for t in range(1,length):
        # Check for collisions between human and agent positions at each step
        agent_next_state = path[t]
        human_state = [human.state for human in simulator.agents]
        simulator._simulate_step()
        for i,human in enumerate(simulator.agents):
            if (human.state == agent_next_state):
                collision_count += 1
                reward -= 2
                collision_step.append(t)
            if t <= length-1:
                if (human.state == agent_state and human_state[i] == agent_next_state):
                    collision_count += 1
                    reward -= 2
                    collision_step.append(t)
        agent_state = agent_next_state
    if collision_count:
        conflict = True
        
    return collision_count, collision_step, collision_position, human_path, conflict, reward
#-------------------------------------------------------------------------------        

def evaluation_a_star(args):
    i,agent_start,agent_goals,obstacles,size,num_humans,human_simulations,budget,alpha,gamma,epsilon,n_episode,human_start,human_goal = args
    env = Env(agent_start, agent_goals, obstacles, size)
    simulator = MultiHumanSimulator(env, human_start, human_goal, n_humans=num_humans)
    start_time = time.time()
    a_star_agent = AStarAgent(env)
    end_time = time.time()
    Time = end_time - start_time
    path = a_star_agent.a_star_search(agent_start,agent_goals)
    collision_count_a_star, collision_step_a_star, collision_postion_a_star, a_star_human_path, conflict, reward_a_star = Evaluation_collision_with_human(path, simulator,agent_goals)
    a_star_success = 1 if (not conflict) and (path[-1] == agent_goals) else 0
    if i % 1 == 0:
        env.create_grid_map('result/mprrt/Map_{}/human_{}/a_star/a_star_path/a_star-path_al_{}_ga_{}_eps_{}_epis_{}_iter={}.pdf'.format(size[0],num_humans,alpha,gamma,epsilon,n_episode,i), path=path, rows=size[0],cols=size[1])
    del env
    return {
        'a_star': (collision_count_a_star, a_star_success, reward_a_star, collision_step_a_star, Time)
    }

def evaluation_mprl(args):
    mul_path = []
    i,agent_start,agent_goals,obstacles,size,num_humans,human_simulations,budget,alpha,gamma,epsilon,n_episode,human_start,human_goal = args
    env = Env(agent_start, agent_goals, obstacles, size)
    agent = QLearningAgent(env,actions=list(range(env.n_actions)),n_episodes=n_episode)
    simulator = MultiHumanSimulator(env, human_start, human_goal, n_humans=num_humans,steps_per_sim=budget)
    start_time = time.time()
    Q, rewards_simulation = agent.q_learning(alpha,gamma,epsilon,agent_start,agent_goals)
    multi_generator = MultiPathGenerator(env, Q, agent_start, agent_goals, N=2000, T=budget)
    
    mul_path_tuple = multi_generator.generate_paths()
    for path in mul_path_tuple:
        mul_path.append(list(path))
    slect_path = SIM(env,mul_path, beta=0.2)

    # Risk calculation and path selection
    all_paths = simulator.simulate(num_sims=human_simulations)
    risk_calc = RiskCalculator(all_paths, total_steps=budget, num_sims=human_simulations, show=False)
    risk_map = risk_calc.calculate_risk(env)
    safe_path = safer_path(slect_path, risk_map)
    end_time = time.time()
    Time = end_time - start_time
        
    # MPRL evaluation
    collision_count, collision_step, collision_postion, human_path, conflict, reward_mprl= Evaluation_collision_with_human(safe_path,simulator,agent_goals)
    mprl_success = 1 if (not conflict) and (safe_path[-1] == agent_goals) else 0
    if i % 1 == 0:
        env.create_grid_map('result/mprrt/Map_{}/human_{}/mprl/mprl_safe/mprl-path_al_{}_ga_{}_eps_{}_epis_{}_iter{}.pdf'.format(size[0],num_humans,alpha,gamma,epsilon,n_episode,i), path=safe_path, rows=size[0],cols=size[1])
    if i % 1 == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(rewards_simulation)), rewards_simulation, '--', linewidth=2)
        plt.title('trian_alpha={}, gamma={}, epsilon={}, episode={}'.format(alpha,gamma,epsilon,n_episode))
        plt.xlabel('episode')
        plt.ylabel('Reward')
        plt.savefig('result/mprrt/Map_{}/human_{}/mprl/mprl_train/train-reward_al_{}_ga_{}_eps_{}_epi_{}_iter_{}.pdf'.format(size[0],num_humans,alpha,gamma,epsilon,n_episode,i),dpi=300,bbox_inches='tight')
        plt.close()
    del env
    return {
        'mprl': (collision_count, mprl_success, reward_mprl, collision_step, Time)
    }

def evaluation_mdp(args):
    i,agent_start,agent_goals,obstacles,size,num_humans,human_simulations,budget,alpha,gamma,epsilon,n_episode,human_start,human_goal = args
    env = Env(agent_start, agent_goals, obstacles, size)
    agent = QLearningAgent(env,actions=list(range(env.n_actions)),n_episodes=n_episode)
    simulator = MultiHumanSimulator(env, human_start, human_goal, n_humans=num_humans)
    start_time = time.time()
    Q_human, reward_with_human = agent.q_learning_with_human(alpha,gamma,epsilon,agent_start,agent_goals,simulator)
    classical_path = path_generator_with_human(env, Q_human,agent_start,agent_goals,budget)
    end_time = time.time()
    Time = end_time - start_time
    collision_count_human, collision_step_human, collision_postion_human, human_path, conflict_human, reward_mdp = Evaluation_collision_with_human(classical_path, simulator,agent_goals)
    mdp_success = 1 if (not conflict_human) and (classical_path[-1] == agent_goals) else 0
    if i % 1 == 0:
        env.create_grid_map('result/mprrt/Map_{}/human_{}/mdp/mdp_path/mdp-path_al_{}_ga_{}_eps_{}_epis_{}_iter_{}.pdf'.format(size[0],num_humans,alpha,gamma,epsilon,n_episode,i), path=classical_path, rows=size[0],cols=size[1])
    if i % 1 == 0:
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(reward_with_human)), reward_with_human, '--', linewidth=2)
        plt.title('trian_alpha={}, gamma={}, epsilon={}, episode={}'.format(alpha,gamma,epsilon,n_episode))
        plt.xlabel('episode')
        plt.ylabel('Reward')
        plt.savefig('result/mprrt/Map_{}/human_{}/mdp/mdp_train/train_reward_al_{}_ga_{}_eps_{}_epis_{}_iter_{}.pdf'.format(size[0],num_humans,alpha,gamma,epsilon,n_episode,i),dpi=300,bbox_inches='tight')
        plt.close()
    del env
    return {
       'mdp': (collision_count_human, mdp_success, reward_mdp, collision_step_human, Time)
    }

def parse_coordinates(arg):
    # Remove parentheses and spaces, split by comma
    parts = arg.replace('(', '').replace(')', '').replace(' ', '').split(',')
    # Convert every two elements to a tuple
    coordinates = [(int(parts[i]), int(parts[i + 1])) for i in range(0, len(parts), 2)] #Combine strings in pairs to form tuple coordinates.
    return coordinates

def save_plot_data_as_txt(x_data,y_data,filename):
    with open(filename, 'w') as f:
        f.write("step\tconflict_number\n")
        for x,y in zip(x_data,y_data):
            f.write(f"{x}\t{y}\n")

def count_txt_files(directory_path):
    """
    Count txt files in specified directory
    Parameters directory_path: Target directory path (string)
    Returns: txt file count
    """
    try:
        # Create Path object and resolve path
        path = Path(directory_path).expanduser().resolve()
        
        # Validate path validity
        if not path.exists():
            print(f"Error: Path '{path}' does not exist")
            return 0
        if not path.is_dir():
            print(f"Error: '{path}' is not a directory")
            return 0

        # Count txt files
        txt_count = 0
        for file in path.iterdir():
            if file.is_file() and file.suffix.lower() == '.txt':
                txt_count += 1

        return txt_count

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return 0

# multi-processing(8 kernel)
def data_processing(name,budget):
    col_number = []
    success_list = []
    reward_list = []
    Time_list = []
    middle_dict = {step:0 for step in range(1,budget+1)}
    if name == 'a_star':
        with multiprocessing.Pool(8) as pool:
            results = list(tqdm(pool.imap_unordered(evaluation_a_star,args),total=simulation_times))
    elif name == 'mprl':
        with multiprocessing.Pool(8) as pool:
            results = list(tqdm(pool.imap_unordered(evaluation_mprl,args),total=simulation_times))
    else :
        with multiprocessing.Pool(8) as pool:
            results = list(tqdm(pool.imap_unordered(evaluation_mdp,args),total=simulation_times))
    for result in results:
        col, success, reward, col_steps, Time = result[name]
        col_number.append(col)
        success_list.append(success)
        reward_list.append(reward)
        Time_list.append(Time)
        if col_steps:
            for step in col_steps:
                middle_dict[step] += 1
    number = count_txt_files('result/mprrt/Map_{}/human_{}/{}/{}_txt'.format(size[0],num_humans,name,name))+1
    filename = 'result/mprrt/Map_{}/human_{}/{}/{}_result/data_{}'.format(size[0],num_humans,name,name,number)
    with open(filename, 'w', encoding='utf-8') as f:
        # Write conflict rate, success rate and reward
        f.write("{}成功率:{}\n".format(name,success_list))
        f.write("{}成功率均值:{}\n".format(name,np.mean(success_list)))
        f.write("{}成功率标准差:{}\n".format(name,np.std(success_list)))
    
        f.write("{}冲突率:{}\n".format(name,col_number))
        f.write("{}冲突率均值:{}\n".format(name,np.mean(col_number)))
        f.write("{}冲突率标准差:{}\n".format(name,np.std(col_number)))  
        
        f.write("{}回报:{}\n".format(name,reward_list))
        f.write("{}回报均值:{}\n".format(name,np.mean(reward_list)))
        f.write("{}回报标准差:{}\n".format(name,np.std(reward_list)))

        f.write("{}时间:{}\n".format(name,Time_list))
        f.write("{}时间均值:{}\n".format(name,np.mean(Time_list)))
        f.write("{}时间标准差:{}\n".format(name,np.std(Time_list)))
    x_values = list(middle_dict.keys())
    y_values = list(middle_dict.values())
    save_plot_data_as_txt(x_values,y_values,'result/mprrt/Map_{}/human_{}/{}/{}_txt/conflict_{}_{}.txt'.format(size[0],num_humans,name,name,name,number))
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.plot(x_values, y_values, color='blue', marker='o', label=name)
    plt.title('conflict')
    plt.xticks(range(0,budget+1,1))
    plt.xlabel('step')
    plt.ylabel('conflict numbers')
    plt.legend()
    plt.savefig('result/mprrt/Map_{}/human_{}/{}/{}_picture/conflict_{}_{}.pdf'.format(size[0],num_humans,name,name,name,number),dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

# --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.7, help='exploration rate')
    parser.add_argument('--episode', type=int, default=4000, help='number of episodes')
    parser.add_argument('--simulation', type=int, default=100, help='number of simulations for evaluation')
    parser.add_argument('--human_simulations', type=int, default=2000, help='the number of human simulations')
    parser.add_argument('--budget', type=int, default=20, help='number of simulations for evaluation')
    parser.add_argument('--num_humans', type=int, default=1, help='the number of human')
    parser.add_argument('--param', type=str, default='yaml/10x10_obst_rrt.yaml', help='the information of map')
    
    args = parser.parse_args()
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    n_episode = args.episode
    simulation_times = args.simulation
    human_simulations = args.human_simulations
    budget = args.budget
    num_humans = args.num_humans
    
    # the YAML file(./yaml/*.yaml) contains the information of map, agent and human.
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    agent_goal = tuple(param['agents']['goal'])
    agent_start = tuple(param['agents']['start'])
    size = tuple(param['map']['dimensions'])
    obstacles =set(param['map']['obstacles'])

    Number_human = len(param['humans'])
    if num_humans > Number_human:
        print('The number of humans in the YAML file is less than the number of humans in the parameter')
        exit()

    human_start = []
    human_goal = []
    for i in range(1,num_humans+1):
        human_start.append(tuple(param['humans']['human{}'.format(i)]['start']))
        human_goal.append(tuple(param['humans']['human{}'.format(i)]['goal']))
     
    print(size)
    print(agent_goal)
    print(human_goal)
    print(human_start)
    
    #parameter
    args = (1,agent_start,agent_goal,obstacles,size,num_humans,human_simulations,budget,alpha,gamma,epsilon,n_episode,human_start,human_goal)
    a_star = evaluation_a_star(args)
    mdp = evaluation_mdp(args)
    mprl = evaluation_mprl(args)
    print("a_star",a_star)
    print("mdp",mdp)
    print("mprl",mprl)    
    
    print('The algorithm is done! Thanks for your patience!')
    