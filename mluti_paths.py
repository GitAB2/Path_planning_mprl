import copy
from itertools import count
import random
from typing import List, Dict, Set, Tuple, Optional
from environment import Env

class MultiPathGenerator:
    def __init__(self,env, Q: Dict[Tuple, Dict[Tuple, float]], s0: Tuple, sg: Tuple, N = 1000, T=20):
        """
        初始化多路径生成器
        
        参数:
            Q: 状态-动作值函数，格式为 {state: {action: value}}
            s0: 初始状态
            sg: 目标状态
            N: 要生成的路径总数
            T: 路径的最大长度限制
        """
        self.env = env
        self.Q = Q
        self.s0 = s0
        self.sg = sg
        self.N = N
        self.T = T
        self.X_bar = []  # 最终路径集合
        self.A_actions = [] #记录最终路径集合采取的动作
        self.X_opt = []  # 最优路径集合
        self.X_s = []    # 其他路径集合
        self.P_list = {}  # 次优锚点集合
        self.B_list = {}  # 当前最优锚点(分支点)集合
        
    def greedy_policy(self, current_state: Tuple, Action=None, Actions=None, Previous_path=None, visited: Optional[Set[Tuple]] = None) -> List[Tuple]:
        """
        贪心策略生成路径
        
        参数:
            current_state: 当前状态
            visited: 已访问的状态集合(避免循环)
            
        返回:
            生成的路径
        """
        if visited is None:
            visited = set()
        
        if Previous_path:
            path_sequence = Previous_path  # 取第一个路径
            cutoff_index = path_sequence.index(current_state)  # 取截止索引
            # 截断路径并添加当前状态
            path = list(path_sequence[:cutoff_index])   
            # 同步截断动作
            actions = list(Actions[:cutoff_index])
            limit_t = cutoff_index
        else:
            path = []
            actions = []
            limit_t = -1
        path.append(current_state)
        visited = set(path)
        t = len(path)-1
       
        while current_state != self.sg and t <= self.T and self.manhattan_distance(current_state,self.sg) <= self.T-t:
            # if current_state not in self.Q or not self.Q[current_state]:
            #     break  # 无可用动作
            if current_state in self.B_list.keys():
                a = Action
                if not self.B_list[current_state]:
                    if a not in self.B_list[current_state]:
                        print('actions:',current_state, a, self.B_list)
                    self.B_list[current_state].remove(a)
                    if not self.B_list[current_state]:
                        del self.B_list[current_state]
            else:
                a = max(self.Q[current_state].keys(), key=lambda a: self.Q[current_state][a])
            actions.append(a)
            next_state = self._apply_action(current_state, a)
            
            if next_state in visited:
                break  # 避免循环
        
            current_state = next_state
            path.append(current_state)
            visited.add(current_state)
            
            t += 1
        # print("pair",path,actions)

        return tuple(path), tuple(actions), limit_t

    def sub_greedy_policy(self, current_state: Tuple, Action=None, Actions=None, Previous_path=None, visited: Optional[Set[Tuple]] = None) -> List[Tuple]:
        """
        贪心策略生成路径
        
        参数:
            current_state: 当前状态
            visited: 已访问的状态集合(避免循环)
            
        返回:
            生成的路径
        """
        if visited is None:
            visited = set()
        
        if Previous_path:
            path_sequence = Previous_path  # 取第一个路径
            if current_state not in path_sequence:
                print(path_sequence)
                print(current_state)
                print(self.B_list)
            cutoff_index = path_sequence.index(current_state)  # 取截止索引
            # 截断路径并添加当前状态
            path = list(path_sequence[:cutoff_index])   
            # 同步截断动作
            actions = list(Actions[:cutoff_index])
            limit_t = cutoff_index
        else:
            path = []
            actions = []
            limit_t = -1
        path.append(current_state)
        visited = set(path)
        t = len(path)-1
       
        while current_state != self.sg and t <= self.T and self.manhattan_distance(current_state,self.sg) <=+self.T-t:
            # if current_state not in self.Q or not self.Q[current_state]:
            #     break  # 无可用动作
            if current_state in self.P_list.keys():
                a = Action
                if not self.P_list[current_state]:
                    if a not in self.P_list[current_state]:
                        print('actions:',current_state, a, self.P_list)
                    self.P_list[current_state].remove(a)
                    if not self.P_list[current_state]:
                        del self.P_list[current_state]
            else:
                all_a = list(self.Q[current_state].keys())
                best_action = [a for a in all_a if abs(self.Q[current_state][a] - max(self.Q[current_state].values())) < 1e-3]
                a_reverse = self.limit_action(actions[-1])
                if a_reverse in best_action: #限制重复的动作，避免重复路径而丢失解
                    best_action.remove(a_reverse)
                if best_action:
                    a = random.choice(best_action)
                else:
                    a = random.choice(list(set(all_a) - set(best_action)))
            actions.append(a)
            next_state = self._apply_action(current_state, a)
            
            if next_state in visited:
                break  # 避免循环
        
            current_state = next_state
            path.append(current_state)
            visited.add(current_state)
            
            t += 1
        # print("pair",path,actions)

        return tuple(path), tuple(actions), limit_t
    
    def _apply_action(self, state: Tuple, action: Tuple) -> Tuple:
        """
        应用动作到状态，得到新状态
        (这里需要根据具体问题实现状态转换逻辑)
        """
        next_state, _ = self.env.step(state, action)
        return next_state
    
    def find_potential_anchors(self, Path: Tuple[Tuple]) -> List[Tuple]:
        """
        在路径中寻找潜在锚点(有多个可选动作的状态)
        """
        anchors = []
        # path = list(Path)[0]  # 不包括最后一个状态
        for state in Path[:-1]:
            if state == self.s0 and len(self.Q[state]) >= 2:
                anchors.append(state)
            else:    
                if state in self.Q and len(self.Q[state]) >= 3:
                    anchors.append(state)
        return anchors
    
    def update_lists(self, Path: Tuple[Tuple], Actions: List[int], limit_t: int):

        """
        更新P_list和B_list
        """
        # 寻找路径中的锚点并添加到B_list
        anchors = self.find_potential_anchors(Path)
        actions = list(Actions)
        self.B_list.clear()
        # print('anchors:',anchors)

        #i == 0 需注意
        for anchor in anchors:
            i = Path.index(anchor)
            if i <= limit_t:
                continue
            max_value = max(self.Q[anchor].values())
            all_a = list(self.Q[anchor].keys())
            # if actions[i] not in all_a:
            #     print('actions:',anchor,actions, path, i, path[i], actions[i])
            all_a.remove(actions[i])
            if i > 0:
                all_a.remove(self.limit_action(actions[i-1]))
                a_list = [a for a in all_a if abs(max_value - self.Q[anchor][a]) < 1e-3]
                random.shuffle(a_list)
                if a_list:
                    self.B_list.update({anchor:a_list})
            else:
                if all_a:
                    self.B_list.update({anchor:all_a})
        # print('B_list:',self.B_list)
            
    def update_lists_suboptimal(self, Path: List[Tuple], Actions: List[int], limit_t: int):

        anchors = self.find_potential_anchors(Path)
        actions = list(Actions)
        self.P_list.clear()

        for anchor in anchors:
            i = Path.index(anchor)
            if i <= limit_t:
                continue
            max_value = max(self.Q[anchor].values())
            all_a = list(self.Q[anchor].keys())
            # if actions[i] not in all_a:
            #     print('actions:',anchor,actions, path, path.index(anchor),path[path.index(anchor)],actions[path.index(anchor)])

            all_a.remove(actions[i])
            if i > 0:
                all_a.remove(self.limit_action(actions[i-1]))
                a_list = [a for a in all_a if abs(max_value - self.Q[anchor][a]) > 1e-3]
                a_list.sort(key=lambda x: self.Q[anchor][x], reverse=True)
                if a_list:
                    self.P_list.update({anchor:a_list})
            else:
                a_list = [a for a in all_a if abs(max_value - self.Q[anchor][a]) > 1e-3]
                if a_list:
                    self.P_list.update({anchor:a_list})     
            

    def limit_action(self, previous_action=None):
        """
        限制动作顺序，避免连续相反的动作
        """
        
        # 定义相反动作的映射
        opposite_actions = {
            0 : 1,
            1 : 0,
            2 : 3,
            3 : 2
        }
        action = opposite_actions.get(previous_action)
        return action

    
    def generate_paths(self) -> List[List[Tuple]]:
        """
        主函数：生成多路径
        
        返回:
            生成的路径集合
        """
        # 1. 生成基础最优路径
        base_path, actions, t = self.greedy_policy(self.s0)
        if not base_path:
            print('have no path')
            return []
        self.X_bar = {}
        self.X_bar.update({
            base_path:{
                'limit_t':t,
                'actions':actions}
                })
        
        next_path = {}
        next_path.update(self.X_bar)
        before_path = {}
        flag = 0

        # 2. 主循环生成多路径
        while len(self.X_bar) < self.N:
            # 2.1 从B_list生成新路径
            before_path = copy.deepcopy(next_path)
            next_path.clear()
            for _,(base,base_dict) in enumerate(before_path.items()):
                actions = base_dict['actions']
                limit_t = base_dict['limit_t']
                self.update_lists(base,actions,limit_t)
                if not self.B_list:
                    continue
                for anchor in self.B_list.keys():
                    for a in self.B_list[anchor]:
                        new_path, all_a, t = self.greedy_policy(anchor, a, Previous_path = base, Actions = actions)
                        if new_path and new_path[-1] == self.sg and len(new_path) <= self.T:
                            next_path.update({new_path:{'limit_t':t,'actions':all_a}})
                            if len(self.X_bar) < self.N:
                                self.X_bar.update({new_path:{'limit_t':t,'actions':all_a}})
                            else:
                                flag = 1
                                break
                    if flag:
                        break
                if flag:
                    break
            if not next_path:
                break

        print(len(self.X_bar))
        first_next_path = {}
        second_next_path = copy.deepcopy(self.X_bar)

        while len(self.X_bar) < self.N:
            # 2.2 从最优路径中寻找潜在锚点生成更多路径
            
            before_path = copy.deepcopy(second_next_path)
            first_next_path.clear()
            second_next_path.clear()
            for _,(base,base_dict) in enumerate(before_path.items()):
                actions = base_dict['actions']
                limit_t = base_dict['limit_t']
                self.update_lists_suboptimal(base, actions, limit_t)
                if not self.P_list:
                    continue
                for anchor in self.P_list.keys():
                    for a in self.P_list[anchor]:
                        first_new_path, all_a, t = self.sub_greedy_policy(anchor, a, Previous_path = base, Actions = actions)
                        if first_new_path and first_new_path[-1] == self.sg and len(first_new_path) <= self.T:
                            first_next_path.update({first_new_path:{'limit_t':t,'actions':all_a}})
                            if len(self.X_bar) < self.N:
                                self.X_bar.update({first_new_path:{'limit_t':t,'actions':all_a}})
                            else:
                                flag = 1
                                break
                    if flag:
                        break
                if flag:
                    break
            if not first_next_path:
                break
            for _, (base, base_dict) in enumerate(first_next_path.items()):
                actions = base_dict['actions']
                limit_t = base_dict['limit_t']
                self.update_lists(base, actions, limit_t)
                if not self.B_list:
                    continue
                for anchor in self.B_list.keys():
                    for a in self.B_list[anchor]:
                        second_new_path, all_a, t = self.greedy_policy(anchor, a, Previous_path = base, Actions = actions)
                        if second_new_path and second_new_path[-1] == self.sg and len(second_new_path) <= self.T:
                            second_next_path.update({second_new_path:{'limit_t':t,'actions':all_a}})
                            if len(self.X_bar) < self.N:
                                self.X_bar.update({second_new_path:{'limit_t':t,'actions':all_a}})
                            else:
                                flag = 1
                                break
                    if flag:
                        break
                if flag:
                    break
            if not second_next_path:
                break
            second_next_path.update(first_next_path)
        
        # print('len:',len(next_path))                           

        return self.X_bar  # 返回不超过N条的路径

    def manhattan_distance(self, state1: Tuple, state2: Tuple) -> int:
        """
        计算曼哈顿距离
        """
        return abs(state1[0] - state2[0]) + abs(state1[1] - state2[1])

# 示例使用
if __name__ == "__main__":
    try:
        # 示例状态-动作值函数
        # 假设状态是(x,y)坐标，动作是移动方向('up', 'down', 'left', 'right')
        Q = {
            (0, 0): { (0, 1): 1.0, (1, 0): 0.9 },  # 从(0,0)可以向上或向右
            (0, 1): { (0, 2): 1.0, (1, 1): 0.8 },  # 从(0,1)可以向上或向右
            (1, 0): { (1, 1): 0.7, (2, 0): 0.6 },  # 从(1,0)可以向上或向右
            (0, 2): { (0, 3): 1.0, (1, 2): 0.5 },  # 目标状态是(0,3)
            (1, 1): { (1, 2): 0.6, (0, 1): 0.4 },  # 可以向上或向左(但向左会循环)
            (2, 0): { (2, 1): 0.5, (1, 0): 0.3 },  # 可以向上或向左
            (1, 2): { (1, 3): 0.7, (0, 2): 0.2 },  # 可以向上或向左
            (2, 1): { (2, 2): 0.4, (1, 1): 0.3 },  # 可以向上或向左
            (1, 3): {},  # 目标状态没有出边
            (2, 2): { (2, 3): 0.3, (1, 2): 0.2 },  # 可以向上或向左
            (0, 3): {},  # 目标状态
            (2, 3): {}   # 另一个可能的目标状态
        }
        
        s0 = (0, 0)  # 初始状态
        sg = (0, 3)  # 目标状态
        N = 10       # 要生成的路径数
        T = 10       # 路径最大长度
        env = Env(s0, sg)
        generator = MultiPathGenerator(env, Q, s0, sg, N, T)
        paths = generator.generate_paths()
        
        print(f"Generated {len(paths)} paths:")
        for i, path in enumerate(paths, 1):
            print(f"Path {i}: {path}")
    except Exception as e:
        print(f"An error occurred: {e}")