import argparse
import numpy as np
import random
import time
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import copy

import tqdm
import yaml
# from main import Evaluation_collision_with_human, MultiHumanSimulator, RiskCalculator, count_txt_files, data_processing, safer_path, save_plot_data_as_txt
from environment import Env

class Node:
    __slots__ = ('x', 'y', 'parent')
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent

    def position(self):
        return (self.x, self.y)
    
class RRTEnvironment(Env):
    """
    继承自 environment.Env 的 RRT 专用环境类
    提供 RRT 算法所需的额外功能
    """
    def __init__(self, agent_start, agent_goals, obstacles, size, headless=False):
        # 调用父类构造函数
        super().__init__(agent_start, agent_goals, obstacles, size, headless=headless)
        
        # # 确保 obstacles 属性已设置
        # if not hasattr(self, 'obstacles'):
        #     self.obstacles = obstacles
        
        # # 确保 able_state 属性已设置
        # if not hasattr(self, 'able_state'):
        #     self.able_state = self.traversable_state()
        
        # 初始化 RRT 特有的区域划分
        self.area1 = []  # 右上
        self.area2 = []  # 右下
        self.area3 = []  # 左上
        self.area4 = []  # 左下
        
        self.visited_area1 = set()
        self.visited_area2 = set()
        self.visited_area3 = set()
        self.visited_area4 = set()
        
        # 初始化区域
        self._initialize_areas()
        
        # 动态区域（用于采样）
        self.dyna_area1 = copy.deepcopy(self.area1)
        self.dyna_area2 = copy.deepcopy(self.area2)
        self.dyna_area3 = copy.deepcopy(self.area3)
        self.dyna_area4 = copy.deepcopy(self.area4)
    
    def _initialize_areas(self):
        """初始化四个象限区域"""
        mid_x = int(self.width * 0.5)
        mid_y = int(self.height * 0.5)
        
        # 右上
        for x in range(mid_x, self.width):
            for y in range(mid_y, self.height):
                if (x, y) not in self.obstacles:
                    self.area1.append((x, y))
        
        # 右下
        for x in range(mid_x, self.width):
            for y in range(0, mid_y):
                if (x, y) not in self.obstacles:
                    self.area2.append((x, y))
        
        # 左上
        for x in range(0, mid_x):
            for y in range(mid_y, self.height):
                if (x, y) not in self.obstacles:
                    self.area3.append((x, y))
        
        # 左下
        for x in range(0, mid_x):
            for y in range(0, mid_y):
                if (x, y) not in self.obstacles:
                    self.area4.append((x, y))
                    
    
    # def reset_areas(self):
    #     """重置动态区域（用于多次路径生成）"""
    #     self.dyna_area1 = copy.deepcopy(self.area1)
    #     self.dyna_area2 = copy.deepcopy(self.area2)
    #     self.dyna_area3 = copy.deepcopy(self.area3)
    #     self.dyna_area4 = copy.deepcopy(self.area4)
        
    #     self.visited_area1.clear()
    #     self.visited_area2.clear()
    #     self.visited_area3.clear()
    #     self.visited_area4.clear()





# class Environment(object):
#     def __init__(self, obstacles, GRID_SIZE):

#         self.obstacles = obstacles
#         self.GRID_SIZE = GRID_SIZE

#         self.area1 = []
#         self.area2 = []
#         self.area3 = []
#         self.area4 = []

#         self.visited_area1 = set()
#         self.visited_area2 = set()
#         self.visited_area3 = set()
#         self.visited_area4 = set()

#         self.initialization()

#         self.dyna_area1 = copy.deepcopy(self.area1)
#         self.dyna_area2 = copy.deepcopy(self.area2)
#         self.dyna_area3 = copy.deepcopy(self.area3)
#         self.dyna_area4 = copy.deepcopy(self.area4)

#     def initialization(self):

#         #右上
#         for x in range(int(self.GRID_SIZE[0]*0.5),self.GRID_SIZE[0]):
#             for y in range(int(self.GRID_SIZE[1]*0.5),self.GRID_SIZE[1]):
#                 if (x,y) not in self.obstacles:
#                     self.area1.append((x,y))

#         #右下
#         for x in range(int(self.GRID_SIZE[0]*0.5), self.GRID_SIZE[0]):
#             for y in range(0, int(self.GRID_SIZE[0]*0.5)):
#                 if (x,y) not in self.obstacles:
#                     self.area2.append((x,y))
#         #左上
#         for x in range(0, int(self.GRID_SIZE[0]*0.5)):
#             for y in range(int(self.GRID_SIZE[0]*0.5), self.GRID_SIZE[1]):
#                 if (x,y) not in self.obstacles:
#                     self.area3.append((x,y))

#         # 左下
#         for x in range(0,int(self.GRID_SIZE[0]*0.5)):
#             for y in range(0,int(self.GRID_SIZE[1]*0.5)):
#                 if (x,y) not in self.obstacles:
#                     self.area4.append((x,y))





def generate_obstacles(GRID_SIZE):
    """生成障碍物并确保多条路径存在"""
    obstacles = set()
    for x in [1,2,3,4,5,6,7,8,11,12,13,14,15,16,17,18]:
        for y in [1,3,5,7,9,12,14,16,18]:
            obstacles.add((x,y))

    return obstacles

def generate_obstacles_10(GRID_SIZE):
    """生成障碍物并确保多条路径存在"""
    obstacles = set()
    for x in [1,3,6,8]:
        for y in [1,2,3,6,7,8]:
            obstacles.add((x,y))

    return obstacles


def count_possible_paths(GRID_SIZE, obstacles, start, goal):
    """BFS计算可达路径数量（简化版）"""
    # 使用简化的连通性检查替代完整路径计数
    if not is_connected(GRID_SIZE, obstacles, start, goal):
        return 0
    
    # 尝试不同路径方向
    path_count = 0
    for bias in ['horizontal', 'vertical', 'mixed']:
        if find_path_with_bias(GRID_SIZE, obstacles, bias, start, goal):
            path_count += 1
    return path_count

def find_path_with_bias(GRID_SIZE, obstacles, bias, start, goal):
    """带方向偏好的路径查找"""
    queue = deque([(start[0], start[1], [])])
    visited = set([start])
    
    while queue:
        x, y, path = queue.popleft()
        new_path = path + [(x, y)]
        
        if (x, y) == goal:
            return new_path
        
        # 根据偏好确定探索顺序
        if bias == 'horizontal':
            moves = [(1,0), (-1,0), (0,1), (0,-1)]  # 优先水平
        elif bias == 'vertical':
            moves = [(0,1), (0,-1), (1,0), (-1,0)]  # 优先垂直
        else:
            moves = [(1,0), (0,1), (-1,0), (0,-1)]  # 混合
        
        for dx, dy in moves:
            nx, ny = x + dx, y + dy
            if (0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1] and
                (nx, ny) not in obstacles and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny, new_path))
    return None

def is_connected(GRID_SIZE, obstacles, start, goal):
    """BFS检查起点到终点的连通性"""
    visited = [[False for _ in range(GRID_SIZE[1])] for _ in range(GRID_SIZE[0])]
    queue = deque([start])
    visited[start[0]][start[1]] = True
    
    while queue:
        x, y = queue.popleft()
        if (x, y) == goal:
            return True
        
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1]:
                if not visited[nx][ny] and (nx, ny) not in obstacles:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
    return False

def rrt(env, start, goal, GRID_SIZE, obstacles, max_iter, sampling_strategy):
    """增强的RRT路径搜索算法"""
    tree = [Node(*start)]


    for i in range(max_iter):
        # 智能采样策略
        #if i % 5 == 0:  # 20%的概率采样目标点

        if sampling_strategy == "quadrant" and random.random()>0.1:

            
            alpha = random.random()

            a1 = 1-len(env.visited_area1)/len(env.area1)
            a2 = 1-len(env.visited_area2)/len(env.area2)
            a3 = 1-len(env.visited_area3)/len(env.area3)
            a4 = 1-len(env.visited_area4)/len(env.area4)
            total_weight = a1 + a2 + a3 + a4
            if total_weight <= 1e-3:
                a1_threshold = 0.25
                a2_threshold = 0.5
                a3_threshold = 0.75
            else:
                a1_threshold = a1/(a1+a2+a3+a4)
                a2_threshold = a1+a2/(a1+a2+a3+a4)
                a3_threshold = a1+a2+a3/(a1+a2+a3+a4)


            if alpha < a1_threshold and env.dyna_area1:
                #quad = (int(GRID_SIZE[0]*0.5),GRID_SIZE[0]-1,int(GRID_SIZE[1]*0.5),GRID_SIZE[1]-1) # 右上
                rand_point = random.choice(env.dyna_area1)
                
            elif alpha < a2_threshold and env.dyna_area2:
                #quad = (int(GRID_SIZE[0]*0.5), GRID_SIZE[0]-1, 0, int(GRID_SIZE[0]*0.5)-1)  # 右下
                #print(alpha,"a1-a4", a1, a2, a3,a4,"a1_threshold~", a1_threshold, a2_threshold,a3_threshold)
                rand_point = random.choice(env.dyna_area2)
                
            elif alpha < a3_threshold and env.dyna_area3:
                #quad = (0, int(GRID_SIZE[0]*0.5)-1, int(GRID_SIZE[0]*0.5), GRID_SIZE[1]-1) # 左上
                rand_point = random.choice(env.dyna_area3)
                
            elif env.dyna_area4:
                #quad = (0,int(GRID_SIZE[0]*0.5)-1,0,int(GRID_SIZE[1]*0.5)-1) # 左下
                rand_point = random.choice(env.dyna_area4)
            else :
                rand_point = goal

        elif sampling_strategy == 'bridge' and random.random()>0.1:

            rand_point = random.choice(tree).position()

        else:

            rand_point = goal
        
        # 寻找最近节点（曼哈顿距离）
        nearest = min(tree, key=lambda n: 
                     abs(n.x - rand_point[0]) + abs(n.y - rand_point[1]))
        
        # 生成候选移动方向
        candidates = []
        for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
            nx, ny = nearest.x + dx, nearest.y + dy
            if (0 <= nx < GRID_SIZE[0] and 0 <= ny < GRID_SIZE[1] and 
                (nx, ny) not in obstacles):
                candidates.append((nx, ny))
        
        if not candidates:
            continue
        
        # 选择最接近采样点的候选
        new_point = min(candidates, key=lambda p: 
                       abs(p[0]-rand_point[0]) + abs(p[1]-rand_point[1]))
        
        # 检查节点是否已存在
        node_exists = any(n.x == new_point[0] and n.y == new_point[1] for n in tree)
        if node_exists:
            continue
            
        new_node = Node(new_point[0], new_point[1], nearest)
        tree.append(new_node)
        
        # 检查是否到达终点
        if new_point == goal:
            path = []
            current = new_node
            while current:
                path.append(current.position())
                current = current.parent

            # 统计经过区域信息
            for item in path:
                if item in env.area1:
                    env.visited_area1.add(item)
                    try:
                        env.dyna_area1.remove(item)
                    except:
                        1
                elif item in env.area2:
                    env.visited_area2.add(item)
                    try:
                        env.dyna_area2.remove(item)
                    except:
                        1
                elif item in env.area3:
                    env.visited_area3.add(item)
                    try:
                        env.dyna_area3.remove(item)
                    except:
                        1
                elif item in env.area4:
                    env.visited_area4.add(item)
                    try:
                        env.dyna_area4.remove(item)
                    except:
                        1
            return list(reversed(path))
    
    return None  # 未找到路径

def jaccard_similarity(path1, path2):## 1: same path, 0: completely different
    """计算两条路径的Jaccard相似度"""
    set1 = set(path1)
    set2 = set(path2)
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

def plot_conflict_rate(data_dict, name, human_number=1):
    for result in data_dict:
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
        # 写入冲突率,成功率和回报
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
    save_plot_data_as_txt(x_values,y_values,'result/mprrt/Map_{}/human_{}/{}/{}_txt/conflict_{}_{}.txt'.format(size[0],human_number,name,name,name,number))

    plt.rcParams['figure.figsize'] = (10, 6)
    plt.plot(x_values, y_values, color='blue', marker='o', label=name)
    plt.title('conflict')
    plt.xticks(range(0,budget+1,1))
    plt.yticks(range(0,10,2))
    plt.xlabel('step')
    plt.ylabel('conflict numbers')
    plt.legend()
    plt.savefig('result/mprrt/Map_{}/human_{}/{}/{}_picture/conflict_{}_{}.pdf'.format(size[0],human_number,name,name,name,number),dpi=300,bbox_inches='tight')
    plt.show(block=False)
    plt.pause(2)
    plt.close()

def collision_human(rrt_env, i, human_start, human_goal, time, human_number=0):
    mrrt = {}
    start_time = time.time()
    simulator = MultiHumanSimulator(rrt_env, human_start[:human_number], human_goal[:human_number], n_humans=human_number)
    all_human_paths = simulator.simulate(num_sims=human_simulations)
    risk_calc = RiskCalculator(all_human_paths, total_steps=budget, num_sims=human_simulations, show=False)
    risk_map = risk_calc.calculate_risk(rrt_env)
    rrt_m_path = safer_path(all_paths, risk_map)
    end_time = time.time()
    Time = time + end_time - start_time
    collision_count_mrrt, collision_step_mrrt, collision_postion_rrt, rrt_human_path, conflict, reward_mrrt = Evaluation_collision_with_human(rrt_m_path, simulator, agent_goal)
    mrrt_success = 1 if (not conflict) and (rrt_m_path[-1] == agent_goal) else 0
    if (i+1) % 20 == 0:
        rrt_env.create_grid_map('result/mprrt/Map_{}/human_{}/mrrt/mrrt_path/mrrt-path_al_{}_ga_{}_eps_{}_epis_{}_iter={}.pdf'.format(size[0], 10, alpha, gamma, epsilon, n_episode, i), path=rrt_m_path, rows=size[0], cols=size[1])
    return {'mrrt': (collision_count_mrrt, mrrt_success, reward_mrrt, collision_step_mrrt, Time)}

# 主程序
if __name__ == "__main__":
    # 参数设置
    GRID_SIZE = (40, 40)
    START = (0, 0)
    GOAL = (39, 39)
    N_PATHS = 400
    MIN_PATHS = 300 # 最小路径数要求
    
    # 生成有效障碍物（确保多条路径存在）
    # obstacles = generate_obstacles(GRID_SIZE)
    # print(f"障碍物数量: {len(obstacles)}")
    # 生成有效障碍物（确保多条路径存在）
    # obstacles = generate_obstacles(GRID_SIZE)
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.7, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
    parser.add_argument('--epsilon', type=float, default=0.7, help='exploration rate')
    parser.add_argument('--episode', type=int, default=4000, help='number of episodes')
    parser.add_argument('--simulation', type=int, default=100, help='number of simulations for evaluation')
    # parser.add_argument('--flag', type=bool, default=False, help='train or test')
    parser.add_argument('--human_simulations', type=int, default=2000, help='the number of human simulations')
    parser.add_argument('--budget', type=int, default=80, help='number of simulations for evaluation') #需要设置
    
    parser.add_argument('--num_humans', type=int, default=10, help='the number of human') #需要设置
    parser.add_argument('--param', type=str, default='yaml/40x40_obst_rrt.yaml', help='the size of map')
    # parser.add_argument('--param', type=str, default='yaml/20x20_obst_rrt.yaml', help='the size of map')
    # parser.add_argument('--param', type=str, default='yaml/40x40_obst_rrt.yaml', help='the size of map')
    

    args = parser.parse_args()
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    n_episode = args.episode
    simulation_times = args.simulation
    human_simulations = args.human_simulations
    budget = args.budget
    num_humans = args.num_humans

    col_number = []
    success_list = []
    reward_list = []
    Time_list = []
    middle_dict = {step:0 for step in range(1,budget+1)}

    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    agent_goal = tuple(param['agents']['goal'])
    agent_start = tuple(param['agents']['start'])
    size = tuple(param['map']['dimensions'])
    obstacles =set(param['map']['obstacles'])
    human_start = []
    human_goal = []
    for i in range(1,num_humans+1):
        human_start.append(tuple(param['humans']['human{}'.format(i)]['start']))
        human_goal.append(tuple(param['humans']['human{}'.format(i)]['goal']))

    rrt_env = RRTEnvironment(agent_start=agent_start, agent_goals=agent_goal, 
                            obstacles=obstacles, size=size)
    
    # 生成多条路径
    mrrt_1 = {}
    mrrt_2 = {}
    mrrt_4 = {}
    mrrt_6 = {}
    mrrt_8 = {}
    mrrt_10 = {}

    name = 'mrrt'
    all_paths = []
    attempt_count = 0
    max_attempts = N_PATHS * 50000  # 大幅增加尝试次数
    #sampling_strategies = ['quadrant', 'bridge','goal']  # 多种采样策略
    sampling_strategies = ['quadrant','goal']  # 多种采样策略
    max_iter = 10000
    
    # 动态调整相似度阈值
    base_similarity_threshold = 0.8
    min_similarity_threshold = 0.75
    # 使用新的 RRTEnvironment 类
    
    for i in tqdm(range(100)):
        start_time = time.time()
        while len(all_paths) < MIN_PATHS and attempt_count < max_attempts and (time.time()-start_time) < 20:
            w_time = time.time()

            strategy = random.choice(sampling_strategies)


            path = rrt(rrt_env, agent_start, agent_goal, size, obstacles, max_iter, strategy)
            
            if path:
                # 动态调整相似度阈值
                current_threshold = max(min_similarity_threshold,  base_similarity_threshold - len(all_paths)*0.02)
                
                # 过滤相似路径
                is_similar = False
                for existing in all_paths:
                    if jaccard_similarity(path, existing) > current_threshold:
                        is_similar = True
                        break
                
                if not is_similar:
                    all_paths.append(path)
                    # print(f"生成路径 {len(all_paths)}: 长度={len(path)-1}, 策略={strategy}", "time cost:", round(time.time()-w_time, 5))
            
            attempt_count += 1
        
        tim = time.time()
        mrrt_human_1 = collision_human(rrt_env, i, human_start, human_goal, tim, human_number=1)
        mrrt_human_2 = collision_human(rrt_env, i, human_start, human_goal, tim, human_number=2)
        mrrt_human_4 = collision_human(rrt_env, i, human_start, human_goal, tim, human_number=4)
        mrrt_human_6 = collision_human(rrt_env, i, human_start, human_goal, tim, human_number=6)
        mrrt_human_8 = collision_human(rrt_env, i, human_start, human_goal, tim, human_number=8)
        mrrt_human_10 = collision_human(rrt_env, i, human_start, human_goal, tim, human_number=10) 
        mrrt_1.update(mrrt_human_1)
        mrrt_2.update(mrrt_human_2)
        mrrt_4.update(mrrt_human_4)
        mrrt_6.update(mrrt_human_6)
        mrrt_8.update(mrrt_human_8)
        mrrt_10.update(mrrt_human_10)


    plot_conflict_rate(mrrt_1, name, human_number=1)
    plot_conflict_rate(mrrt_2, name, human_number=2)
    plot_conflict_rate(mrrt_4, name, human_number=4)
    plot_conflict_rate(mrrt_6, name, human_number=6)
    plot_conflict_rate(mrrt_8, name, human_number=8)
    plot_conflict_rate(mrrt_10, name, human_number=10)    

    # print("total time", round(time.time()-start_time,2))
    # # 按路径长度排序
    # all_paths.sort(key=len)
        
    # 可视化
    # plt.figure(figsize=(12, 10))
    # ax = plt.gca()
    
    # # 设置网格
    # for i in range(GRID_SIZE[0] + 1):
    #     ax.axvline(i, color='gray', linestyle='-', alpha=0.3)
    # for j in range(GRID_SIZE[1] + 1):
    #     ax.axhline(j, color='gray', linestyle='-', alpha=0.3)
    
    # # 绘制障碍物
    # for obs in obstacles:
    #     rect = patches.Rectangle((obs[0], obs[1]), 1, 1, 
    #                             linewidth=1, edgecolor='black',
    #                             facecolor='black', alpha=0.7)
    #     ax.add_patch(rect)
    
    # # 绘制路径
    # colors = plt.cm.jet(np.linspace(0, 1, len(all_paths)))
    # for i, path in enumerate(all_paths):
    #     xs, ys = zip(*path)
    #     # 绘制路径线
    #     plt.plot([x + 0.5 for x in xs], [y + 0.5 for y in ys], 
    #              marker='o', markersize=4, linewidth=2, 
    #              color=colors[i], label=f'Path {i+1} (Len={len(path)-1})')
    #     # 绘制路径点
    #     for x, y in path:
    #         if (x, y) != agent_start and (x, y) != agent_goal:
    #             plt.plot(x + 0.5, y + 0.5, 'o', color=colors[i], markersize=4)
    
    # # 标记起点终点
    # plt.plot(agent_start[0] + 0.5, agent_start[1] + 0.5, 'gs', markersize=12, label='Start')
    # plt.plot(agent_goal[0] + 0.5, agent_goal[1] + 0.5, 'rs', markersize=12, label='Goal')
    
    # plt.title(f'RRT Path Planning - {len(all_paths)} Diverse Paths')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.xticks(range(GRID_SIZE[0] + 1))
    # plt.yticks(range(GRID_SIZE[1] + 1))
    # plt.grid(True)
    # plt.xlim(0, GRID_SIZE[0])
    # plt.ylim(0, GRID_SIZE[1])
    # # plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    # plt.show()
    
    # 输出最优路径
    #print("\nTop 3 Optimal Paths:")
    #for i, path in enumerate(all_paths[:3]):
    #    print(f"{i+1}. Length={len(path)-1}: {path}")
    
    ## 输出所有路径长度
    #print("\nAll Path Lengths:")
    #for i, path in enumerate(all_paths):
    #    print(f"Path {i+1}: {len(path)-1} steps")


# 使用示例：如何在其他文件中使用 RRTEnvironment
def example_usage():
    """
    示例：如何在其他文件中使用 RRTEnvironment 类
    
    使用方法：
    from mrrt import RRTEnvironment
    
    # 创建环境
    env = RRTEnvironment(
        agent_start=(0, 0),
        agent_goals=(9, 9),
        obstacles={(1, 1), (2, 2), (3, 3)},
        size=(10, 10),
        headless=True
    )
    
    # 使用环境进行路径规划
    path = rrt(env, (0, 0), (9, 9), (10, 10), env.obstacles, 1000, 'goal')
    
    # 重置区域（用于多次路径生成）
    env.reset_areas()
    
    # 访问环境属性
    print(f"可通行状态数量: {len(env.able_state)}")
    print(f"障碍物数量: {len(env.obstacles)}")
    print(f"地图尺寸: {env.size}")
    """
    pass