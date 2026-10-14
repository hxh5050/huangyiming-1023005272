from __future__ import annotations

import heapq
import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 解决Matplotlib中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

Coord = Tuple[int, int]


class GridMap:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        for r, row in enumerate(grid):
            if len(row) != self.cols:
                raise ValueError("地图每一行的长度必须一致")

        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] not in (0, 1):
                    raise ValueError("地图只能包含 0 或 1")

    def in_bounds(self, p: Coord) -> bool:
        r, c = p
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, p: Coord) -> bool:
        r, c = p
        return self.in_bounds(p) and self.grid[r][c] == 0

    def neighbors(self, p: Coord, connectivity: int = 4,
                  allow_corner_cutting: bool = False):
        r, c = p

        if connectivity == 4:
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        elif connectivity == 8:
            deltas = [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)
            ]
        else:
            raise ValueError("connectivity 只能是 4 或 8")

        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)

            if not self.is_free(nxt):
                continue

            if connectivity == 8 and dr != 0 and dc != 0 and not allow_corner_cutting:
                if not (self.is_free((r + dr, c)) and self.is_free((r, c + dc))):
                    continue

            yield nxt, dr, dc

    def move_cost(self, from_p: Coord, to_p: Coord) -> float:
        dr = abs(to_p[0] - from_p[0])
        dc = abs(to_p[1] - from_p[1])

        if dr + dc == 1:
            return 1.0
        elif dr == 1 and dc == 1:
            return math.sqrt(2.0)
        else:
            return math.hypot(dr, dc)


class AStarPlanner:
    def __init__(self,
                 grid_map: GridMap,
                 heuristic: str = "manhattan",
                 connectivity: int = 4,
                 weight: float = 1.0,
                 allow_corner_cutting: bool = False):
        self.map = grid_map
        self.heuristic_name = heuristic.lower()
        self.connectivity = connectivity
        self.weight = weight
        self.allow_corner_cutting = allow_corner_cutting

    def heuristic(self, a: Coord, b: Coord) -> float:
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])

        if self.heuristic_name == "manhattan":
            return dx + dy
        elif self.heuristic_name == "euclidean":
            return math.hypot(dx, dy)
        elif self.heuristic_name == "chebyshev":
            return max(dx, dy)
        elif self.heuristic_name == "octile":
            return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
        elif self.heuristic_name == "zero":
            return 0.0
        else:
            raise ValueError(
                "不支持的启发式函数，可选：manhattan, euclidean, "
                "chebyshev, octile, zero"
            )

    def search(self, start: Coord, goal: Coord):
        if not self.map.is_free(start):
            raise ValueError("起点不可通行或越界")
        if not self.map.is_free(goal):
            raise ValueError("目标点不可通行或越界")

        open_heap = []
        counter = 0

        g_score = {start: 0.0}
        f_score = {start: self.weight * self.heuristic(start, goal)}

        heapq.heappush(open_heap, (f_score[start], counter, start))

        came_from = {}
        closed = set()
        visited_count = 0

        while open_heap:
            f, _, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current == goal:
                path = self._reconstruct_path(came_from, current)
                return path, g_score[current], visited_count, closed

            closed.add(current)
            visited_count += 1

            for nxt, dr, dc in self.map.neighbors(
                    current,
                    connectivity=self.connectivity,
                    allow_corner_cutting=self.allow_corner_cutting
            ):
                if nxt in closed:
                    continue

                tentative_g = g_score[current] + self.map.move_cost(current, nxt)

                if tentative_g < g_score.get(nxt, math.inf):
                    came_from[nxt] = current
                    g_score[nxt] = tentative_g

                    f_new = tentative_g + self.weight * self.heuristic(nxt, goal)
                    f_score[nxt] = f_new

                    counter += 1
                    heapq.heappush(open_heap, (f_new, counter, nxt))

        return [], math.inf, visited_count, closed

    def _reconstruct_path(self, came_from: dict, current: Coord) -> List[Coord]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def a_star_search(grid: List[List[int]],
                  start: Coord,
                  goal: Coord,
                  heuristic: str = "manhattan",
                  connectivity: int = 4,
                  weight: float = 1.0,
                  allow_corner_cutting: bool = False):
    gmap = GridMap(grid)
    planner = AStarPlanner(
        gmap,
        heuristic=heuristic,
        connectivity=connectivity,
        weight=weight,
        allow_corner_cutting=allow_corner_cutting
    )
    path, cost, visited_count, _ = planner.search(start, goal)
    return path, cost, visited_count


def create_demo_map() -> List[List[int]]:
    """15×15 示例地图。0 可通行，1 障碍。"""
    return [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]


def print_grid_with_path(grid: List[List[int]],
                         path: List[Coord],
                         start: Coord,
                         goal: Coord):
    """终端字符画输出（保留原有功能）"""
    path_set = set(path)

    for r, row in enumerate(grid):
        line = []
        for c, val in enumerate(row):
            p = (r, c)

            if p == start:
                line.append("S")
            elif p == goal:
                line.append("G")
            elif p in path_set:
                line.append("*")
            elif val == 1:
                line.append("#")
            else:
                line.append(".")
        print(" ".join(line))


def visualize_path(grid: List[List[int]],
                   path: List[Coord],
                   start: Coord,
                   goal: Coord,
                   title: str = "A* 路径规划结果"):
    """使用 Matplotlib 进行图形化可视化"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制栅格地图：0 白色，1 黑色
    ax.imshow(grid, cmap='binary', origin='upper')

    # 绘制网格线
    ax.set_xticks([x - 0.5 for x in range(len(grid[0]) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(grid) + 1)], minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.tick_params(which='minor', size=0)

    # 设置主刻度
    ax.set_xticks(range(len(grid[0])))
    ax.set_yticks(range(len(grid)))
    ax.set_xticklabels(range(len(grid[0])))
    ax.set_yticklabels(range(len(grid)))

    # 绘制路径
    if path:
        # 注意：path 是 (r, c)，绘图时 x=c, y=r
        px = [c for r, c in path]
        py = [r for r, c in path]
        ax.plot(px, py, color='blue', linewidth=2, label='规划路径')
        ax.scatter(px, py, color='blue', s=15, zorder=3)

    # 绘制起点和终点
    ax.scatter(start[1], start[0], color='green', s=150, marker='o', label='起点 (S)', zorder=4)
    ax.scatter(goal[1], goal[0], color='red', s=150, marker='*', label='终点 (G)', zorder=4)

    # 标注 S 和 G
    ax.text(start[1], start[0], 'S', color='white', ha='center', va='center', fontsize=12, fontweight='bold', zorder=5)
    ax.text(goal[1], goal[0], 'G', color='white', ha='center', va='center', fontsize=12, fontweight='bold', zorder=5)

    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # 调整布局并显示（注意：plt.show() 会阻塞程序，直到关闭窗口）
    plt.tight_layout()
    plt.show()


def main():
    grid = create_demo_map()
    start = (0, 0)
    goal = (14, 14)

    # 四种启发式函数（加入 zero 以便观察盲目搜索的效果）
    heuristics = ["manhattan", "euclidean", "chebyshev", "octile", "zero"]
    # 两种邻域
    connectivities = [4, 8]

    weight = 1.0
    allow_corner_cutting = False

    print("输入地图：")
    for row in grid:
        print(" ".join(str(v) for v in row))
    print("=" * 70)


    summary = []  # (connectivity, heuristic, 节点数, 代价, 扩展节点数)

    for connectivity in connectivities:
        for heuristic in heuristics:
            if heuristic == "manhattan" and connectivity == 8:
                continue
            if heuristic == "octile" and connectivity == 4:
                continue

            print("\n" + "#" * 70)
            print(f"# 配置：邻域={connectivity}  启发式={heuristic}  "
                  f"权重={weight}  禁止切角={not allow_corner_cutting}")
            print("#" * 70)

            path, cost, visited = a_star_search(
                grid,
                start,
                goal,
                heuristic=heuristic,
                connectivity=connectivity,
                weight=weight,
                allow_corner_cutting=allow_corner_cutting
            )

            if path:
                print(f"路径节点数：{len(path)}")
                print(f"路径总代价：{cost:.3f}")
                print(f"扩展节点数：{visited}")

                # 终端字符画展示
                print("\n路径示意图（终端）：")
                print_grid_with_path(grid, path, start, goal)

                # 图形化可视化展示（每次弹出窗口）
                title = f"A* Path (Neighbors={connectivity}, Heuristic={heuristic})\nCost={cost:.2f}, Visited={visited}"
                visualize_path(grid, path, start, goal, title)

                summary.append((connectivity, heuristic,
                                len(path), cost, visited))
            else:
                print("未找到可行路径")
                summary.append((connectivity, heuristic, 0, math.inf, visited))

    # 汇总对比表
    print("\n" + "=" * 70)
    print("汇总对比（权重=1.0，禁止切角）：")
    print("-" * 70)
    print(f"{'邻域':<6}{'启发式':<12}{'节点数':<10}{'路径代价':<14}{'扩展节点数':<10}")
    print("-" * 70)
    for connectivity, heuristic, n_nodes, cost, visited in summary:
        cost_str = f"{cost:.3f}" if math.isfinite(cost) else "无解"
        print(f"{connectivity:<8}{heuristic:<14}{n_nodes:<12}"
              f"{cost_str:<16}{visited:<12}")
    print("-" * 70)


if __name__ == "__main__":
    main()