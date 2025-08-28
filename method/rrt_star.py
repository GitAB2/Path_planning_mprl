import argparse
import math
import random
from typing import List, Tuple, Optional

import yaml


GridPos = Tuple[int, int]


class _Node:
    def __init__(self, pos: GridPos, parent: Optional[int], cost: float) -> None:
        self.pos = pos
        self.parent = parent
        self.cost = cost


def _euclidean(a: GridPos, b: GridPos) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _manhattan(a: GridPos, b: GridPos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _in_bounds(pos: GridPos, width: int, height: int) -> bool:
    return 0 <= pos[0] < width and 0 <= pos[1] < height


def _is_free(pos: GridPos, obstacles: set) -> bool:
    return pos not in obstacles


def _neighbors4(pos: GridPos) -> List[GridPos]:
    x, y = pos
    return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]


def _steer_4dir(from_pos: GridPos, to_pos: GridPos, obstacles: set, width: int, height: int) -> Optional[GridPos]:
    """
    从 from_pos 朝 to_pos 沿4邻域前进一步（贪心减少欧氏距离），返回下一个网格；若被阻挡返回 None。
    """
    candidates = _neighbors4(from_pos)
    # 仅保留边界内且可通行的候选
    valid = [p for p in candidates if _in_bounds(p, width, height) and _is_free(p, obstacles)]
    if not valid:
        return None
    # 无噪声：在最小距离候选中均匀随机打破平局（仅上下左右四方向）
    distances = [_manhattan(p, to_pos) for p in valid]
    # distances = [_euclidean(p, to_pos) for p in valid]
    min_d = min(distances)
    best_list = [p for p, d in zip(valid, distances) if d == min_d]
    return random.choice(best_list)


def _line_free(a: GridPos, b: GridPos, obstacles: set, width: int, height: int) -> bool:
    """
    简单的线可行性检测：在4连通网格上，用贪心步进从 a 走到 b，
    若任一步不可达则视为碰撞。用于近邻连边与重连判定。
    """
    current = a
    max_steps = width * height  # 上界，避免死循环
    steps = 0
    while current != b and steps <= max_steps:
        nxt = _steer_4dir(current, b, obstacles, width, height)
        if nxt is None:
            return False
        current = nxt
        steps += 1
    return current == b


def _path_between(a: GridPos, b: GridPos, obstacles: set, width: int, height: int) -> Optional[List[GridPos]]:
    """
    生成从 a 到 b 的逐格路径（仅上下左右）。若无法到达返回 None。
    使用与 _line_free 相同的贪心步进，确保不穿越障碍。
    """
    path = [a]
    current = a
    max_steps = width * height
    steps = 0
    while current != b and steps <= max_steps:
        nxt = _steer_4dir(current, b, obstacles, width, height)
        if nxt is None:
            return None
        path.append(nxt)
        current = nxt
        steps += 1
    return path if current == b else None


def rrt_star(
    env,
    start: GridPos,
    goal: GridPos,
    max_iter: int = 5000,
    goal_sample_rate: float = 0.05,
    rewire_radius: float = 3.0,
    seed: Optional[int] = None,
) -> List[GridPos]:
    """
    基于当前 Env 网格环境的 RRT* 单路径规划（4连通离散栅格）。

    参数：
    - env: 环境对象（需包含 width/height/obstacles）
    - start, goal: 起点/终点（网格坐标，(x,y)）
    - max_iter: 最大迭代次数
    - goal_sample_rate: 以此概率直接采样目标，促进收敛
    - rewire_radius: 重连半径（欧氏距离）
    - seed: 随机种子（可选）

    返回：
    - 路径列表 [ (x0,y0), (x1,y1), ... ]；若失败返回空列表
    """
    if seed is not None:
        random.seed(seed)

    width, height = env.width, env.height
    obstacles = set(env.obstacles)

    # 快速检查
    if not _in_bounds(start, width, height) or not _in_bounds(goal, width, height):
        return []
    if not _is_free(start, obstacles) or not _is_free(goal, obstacles):
        return []
    if start == goal:
        return [start]

    nodes: List[_Node] = [_Node(start, parent=None, cost=0.0)]

    def nearest_index(sample: GridPos) -> int:
        best_i, best_d = 0, float('inf')
        for i, n in enumerate(nodes):
            d = _manhattan(n.pos, sample)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def near_indices(center_pos: GridPos, radius: float) -> List[int]:
        return [i for i, n in enumerate(nodes) if _manhattan(n.pos, center_pos) <= radius]

    def reconstruct_path(goal_index: int) -> List[GridPos]:
        coarse: List[GridPos] = []
        cur = goal_index
        while cur is not None:
            coarse.append(nodes[cur].pos)
            cur = nodes[cur].parent
        coarse.reverse()
        # 将树中相邻节点间的“直线”边展开为逐格路径，确保仅4连通且不穿障碍
        full: List[GridPos] = []
        for i in range(len(coarse) - 1):
            a, b = coarse[i], coarse[i + 1]
            segment = _path_between(a, b, obstacles, width, height)
            if segment is None:
                return []
            if i == 0:
                full.extend(segment)
            else:
                # 避免重复拼接相邻段的起点
                full.extend(segment[1:])
        return full if full else coarse

    best_goal_index: Optional[int] = None

    for _ in range(max_iter):
        # 采样
        if random.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (random.randrange(0, width), random.randrange(0, height))
            if not _is_free(sample, obstacles):
                continue

        # 最近邻
        idx_near = nearest_index(sample)
        near_pos = nodes[idx_near].pos

        # steer：朝采样点前进一步（4连通）
        new_pos = _steer_4dir(near_pos, sample, obstacles, width, height)
        if new_pos is None:
            continue
        if new_pos == near_pos:
            continue

        # 若边连通性检查不通过则跳过
        if not _line_free(near_pos, new_pos, obstacles, width, height):
            continue

        # 选择最优父节点（在半径内、可连通且总代价最小）
        candidate_parent = idx_near
        candidate_cost = nodes[idx_near].cost + 1.0  # 单步代价=1（4连通）
        for j in near_indices(new_pos, rewire_radius):
            if _line_free(nodes[j].pos, new_pos, obstacles, width, height):
                # 使用网格步长近似代价（曼哈顿距离）
                new_cost = nodes[j].cost + (abs(nodes[j].pos[0]-new_pos[0]) + abs(nodes[j].pos[1]-new_pos[1]))
                if new_cost < candidate_cost:
                    candidate_parent = j
                    candidate_cost = new_cost

        nodes.append(_Node(new_pos, parent=candidate_parent, cost=candidate_cost))
        new_index = len(nodes) - 1

        # 重连（降低周围节点代价）
        for j in near_indices(new_pos, rewire_radius):
            if j == new_index:
                continue
            if not _line_free(new_pos, nodes[j].pos, obstacles, width, height):
                continue
            new_cost_via_new = nodes[new_index].cost + (abs(new_pos[0]-nodes[j].pos[0]) + abs(new_pos[1]-nodes[j].pos[1]))
            if new_cost_via_new + 1e-9 < nodes[j].cost:
                nodes[j].parent = new_index
                nodes[j].cost = new_cost_via_new

        # 目标检测：若可从新节点直接连到目标，则加入目标节点
        if _line_free(new_pos, goal, obstacles, width, height):
            goal_cost = nodes[new_index].cost + (abs(new_pos[0]-goal[0]) + abs(new_pos[1]-goal[1]))
            nodes.append(_Node(goal, parent=new_index, cost=goal_cost))
            best_goal_index = len(nodes) - 1
            break

    # 若循环中未直接连接目标，尝试从现有节点中选择一个可直连目标的最优节点
    if best_goal_index is None:
        best_cost = float('inf')
        best_index = None
        for i, n in enumerate(nodes):
            if _line_free(n.pos, goal, obstacles, width, height):
                c = n.cost + (abs(n.pos[0]-goal[0]) + abs(n.pos[1]-goal[1]))
                if c < best_cost:
                    best_cost = c
                    best_index = i
        if best_index is not None:
            nodes.append(_Node(goal, parent=best_index, cost=best_cost))
            best_goal_index = len(nodes) - 1

    if best_goal_index is None:
        return []

    path = reconstruct_path(best_goal_index)
    return path


def plan_rrt_star_for_env(env, max_iter: int = 5000, goal_sample_rate: float = 0.05, rewire_radius: float = 3.0, seed: Optional[int] = None) -> List[GridPos]:
    """
    便捷封装：直接使用 env.start 与 env.goal[0] 进行规划。
    """
    start = env.start
    goal = env.goal[0] if isinstance(env.goal, (list, tuple)) else env.goal
    return rrt_star(env, start, goal, max_iter=max_iter, goal_sample_rate=goal_sample_rate, rewire_radius=rewire_radius, seed=seed)


if __name__ == "__main__":
    # 简单测试：在二维网格上生成从起点到目标的路径
    try:
        from environment import Env
    except Exception as e:
        print("导入 Env 失败，请在项目根目录运行该脚本。错误:", e)
        raise

    # 读取 10x10 地图参数（含起点、终点、障碍）
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', type=str, default='yaml/20x20_obst_rrt.yaml', help='yaml 参数文件（10x10 地图）')
    args = parser.parse_args()
    with open(args.param, 'r') as param_file:
        try:
            param = yaml.load(param_file, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    size = tuple(param['map']['dimensions']) if 'dimensions' in param['map'] else (10, 10)
    start = tuple(param['agents']['start'])
    goal = tuple(param['agents']['goal'])
    obstacles = set(param['map']['obstacles'])

    # 创建无界面环境（10x10）
    env = Env(agent_start=start, agent_goals=goal, obstacles=obstacles, size=size, headless=False)
    # 适配无GUI模式下未初始化的属性
    if not hasattr(env, 'obstacles') or env.obstacles is None:
        env.obstacles = obstacles
    if not hasattr(env, 'able_state') or not getattr(env, 'able_state', None):
        env.able_state = env.traversable_state()

    # 不固定随机种子，保证每次运行的路径具有随机性
    for i in range(50):
        path = rrt_star(env, start, goal, max_iter=5000, goal_sample_rate=0.1, rewire_radius=3.0)
        if not path:
            print("未找到路径")
        else:
            print("找到路径，长度=", len(path))
            # print(path)
            # 若安装了 reportlab，可导出 PDF 地图
            try:
                env.create_grid_map("rrt_star_picture/rrt_star_test_path_{}.pdf".format(i), path=path, rows=size[0], cols=size[1])

                print("已导出: rrt_star_test_path.pdf")
            except Exception as _:
                pass


