"""
algorithms.py —— 校园导航系统核心算法
包含：
  - Dijkstra  最短路径
  - DFS       深度优先遍历
  - BFS       广度优先遍历
  - 连通性检查
所有函数只依赖 Graph 对象，不修改图结构。
"""

import heapq
from collections import deque
from graph import Graph


# ------------------------------------------------------------------ #
#  Dijkstra 最短路径
# ------------------------------------------------------------------ #

def dijkstra(graph: Graph, start: str, end: str) -> tuple[list[str], float] | None:
    """
    用 Dijkstra 算法求两栋建筑之间的最短路径。

    参数：
        graph : Graph 对象
        start : 起点建筑名称
        end   : 终点建筑名称

    返回：
        (path, distance)
            path     : 从 start 到 end 的建筑名称列表，含首尾
                       例：["主楼", "教一楼", "图书馆"]
            distance : 最短总距离（米）
        None —— 起点/终点不存在，或两点不连通

    算法复杂度：O((V + E) log V)，使用最小堆优化
    """
    # 合法性检查
    if not graph.has_node(start) or not graph.has_node(end):
        return None

    # 起终点相同，直接返回
    if start == end:
        return ([start], 0.0)

    # dist[v] = 目前已知从 start 到 v 的最短距离
    dist = {node: float("inf") for node in graph.all_nodes()}
    dist[start] = 0.0

    # prev[v] = 最短路径中 v 的前驱节点（用于回溯路径）
    prev: dict[str, str | None] = {node: None for node in graph.all_nodes()}

    # 最小堆：(当前距离, 节点名称)
    heap = [(0.0, start)]

    # 已确定最短距离的节点集合
    visited: set[str] = set()

    while heap:
        cur_dist, cur_node = heapq.heappop(heap)

        # 跳过已处理的节点（堆中可能存在旧的、更大的距离）
        if cur_node in visited:
            continue
        visited.add(cur_node)

        # 找到终点，提前退出
        if cur_node == end:
            break

        # 松弛相邻边
        for neighbor, weight in graph.neighbors(cur_node).items():
            if neighbor in visited:
                continue
            new_dist = cur_dist + weight
            if new_dist < dist[neighbor]:
                dist[neighbor] = new_dist
                prev[neighbor] = cur_node
                heapq.heappush(heap, (new_dist, neighbor))

    # 终点不可达
    if dist[end] == float("inf"):
        return None

    # 回溯路径
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return (path, dist[end])


def dijkstra_description(graph: Graph, start: str, end: str) -> str:
    """
    返回最短路径的文字描述，方便直接打印或显示。

    示例输出：
        从【主楼】到【图书馆】的最短路径：
        主楼 → 教一楼 → 图书馆
        总距离：330 米，共经过 2 段道路
    """
    result = dijkstra(graph, start, end)
    if result is None:
        return f"【{start}】与【{end}】之间不存在可达路径。"

    path, distance = result
    path_str = " → ".join(path)
    segments = len(path) - 1
    return (
        f"从【{start}】到【{end}】的最短路径：\n"
        f"  {path_str}\n"
        f"  总距离：{distance:.0f} 米，共经过 {segments} 段道路"
    )


# ------------------------------------------------------------------ #
#  DFS 深度优先遍历
# ------------------------------------------------------------------ #

def dfs(graph: Graph, start: str) -> list[str] | None:
    """
    从指定建筑出发，深度优先遍历所有可达建筑。

    参数：
        graph : Graph 对象
        start : 起点建筑名称

    返回：
        按访问顺序排列的建筑名称列表
        None —— 起点不存在

    特点：沿一条路走到底，再回溯（栈/递归实现）
    """
    if not graph.has_node(start):
        return None

    visited: list[str] = []
    visited_set: set[str] = set()

    def _dfs_recursive(node: str):
        visited_set.add(node)
        visited.append(node)
        for neighbor in sorted(graph.neighbors(node).keys()):  # 排序保证结果稳定
            if neighbor not in visited_set:
                _dfs_recursive(neighbor)

    _dfs_recursive(start)
    return visited


def dfs_iterative(graph: Graph, start: str) -> list[str] | None:
    """
    DFS 迭代版本（使用显式栈，避免递归深度限制）。
    结果与递归版略有不同（邻居入栈顺序相反），但同样合法。

    参数/返回值同 dfs()
    """
    if not graph.has_node(start):
        return None

    visited: list[str] = []
    visited_set: set[str] = set()
    stack = [start]

    while stack:
        node = stack.pop()
        if node in visited_set:
            continue
        visited_set.add(node)
        visited.append(node)
        # 邻居逆序压栈，使弹出顺序与字典序一致
        for neighbor in sorted(graph.neighbors(node).keys(), reverse=True):
            if neighbor not in visited_set:
                stack.append(neighbor)

    return visited


# ------------------------------------------------------------------ #
#  BFS 广度优先遍历
# ------------------------------------------------------------------ #

def bfs(graph: Graph, start: str) -> list[str] | None:
    """
    从指定建筑出发，广度优先遍历所有可达建筑。

    参数：
        graph : Graph 对象
        start : 起点建筑名称

    返回：
        按访问顺序（层级顺序）排列的建筑名称列表
        None —— 起点不存在

    特点：一圈一圈向外扩展（队列实现），适合找"跳数最少"路径
    """
    if not graph.has_node(start):
        return None

    visited: list[str] = []
    visited_set: set[str] = {start}
    queue: deque[str] = deque([start])

    while queue:
        node = queue.popleft()
        visited.append(node)
        for neighbor in sorted(graph.neighbors(node).keys()):
            if neighbor not in visited_set:
                visited_set.add(neighbor)
                queue.append(neighbor)

    return visited


def bfs_with_levels(graph: Graph, start: str) -> list[list[str]] | None:
    """
    BFS 分层版本，返回每一层的建筑列表。

    返回：
        [[第0层], [第1层], [第2层], ...]
        第0层即起点本身，第1层是直接相邻的建筑，以此类推
        None —— 起点不存在

    示例：
        [["主楼"], ["教一楼", "科学会堂"], ["教二楼", "图书馆"]]
    """
    if not graph.has_node(start):
        return None

    levels: list[list[str]] = []
    visited_set: set[str] = {start}
    current_level = [start]

    while current_level:
        levels.append(current_level)
        next_level = []
        for node in current_level:
            for neighbor in sorted(graph.neighbors(node).keys()):
                if neighbor not in visited_set:
                    visited_set.add(neighbor)
                    next_level.append(neighbor)
        current_level = next_level

    return levels


# ------------------------------------------------------------------ #
#  连通性工具
# ------------------------------------------------------------------ #

def is_connected(graph: Graph) -> bool:
    """
    判断图是否整体连通（任意两点之间均可达）。

    返回：
        True  —— 全图连通
        False —— 存在孤立建筑或不连通子图
    """
    nodes = graph.all_nodes()
    if not nodes:
        return True   # 空图视为连通
    visited = bfs(graph, nodes[0])
    return len(visited) == len(nodes)


def connected_components(graph: Graph) -> list[list[str]]:
    """
    找出图中所有连通分量（不连通时有多个子图）。

    返回：
        [[分量1的建筑列表], [分量2的建筑列表], ...]
        若全图连通，返回只含一个元素的列表
    """
    unvisited = set(graph.all_nodes())
    components = []

    while unvisited:
        start = next(iter(unvisited))
        component = bfs(graph, start)
        components.append(component)
        unvisited -= set(component)

    return components


# ------------------------------------------------------------------ #
#  简单测试（直接运行此文件时执行）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # 构建测试图
    g = Graph()
    buildings = [
        ("主楼",    400, 500),
        ("教一楼",  320, 460),
        ("教二楼",  320, 600),
        ("教三楼",  200, 600),
        ("教四楼",  200, 460),
        ("图书馆",  350, 300),
        ("科学会堂", 530, 530),
        ("体育馆",  620, 460),
    ]
    for name, x, y in buildings:
        g.add_node(name, x, y)

    edges = [
        ("主楼",   "教一楼",   150),
        ("主楼",   "科学会堂", 130),
        ("主楼",   "图书馆",   200),
        ("教一楼", "教二楼",   120),
        ("教一楼", "教四楼",   160),
        ("教一楼", "图书馆",   180),
        ("教二楼", "教三楼",   100),
        ("教三楼", "教四楼",   130),
        ("科学会堂","体育馆",  100),
    ]
    for u, v, d in edges:
        g.add_edge(u, v, d)

    # ---- Dijkstra ----
    print("=" * 45)
    print("【最短路径】")
    print(dijkstra_description(g, "教三楼", "体育馆"))
    print()
    print(dijkstra_description(g, "主楼", "教三楼"))

    # ---- DFS ----
    print("\n" + "=" * 45)
    print("【DFS 深度优先遍历（从主楼出发）】")
    print(" → ".join(dfs(g, "主楼")))

    # ---- BFS ----
    print("\n" + "=" * 45)
    print("【BFS 广度优先遍历（从主楼出发）】")
    print(" → ".join(bfs(g, "主楼")))

    print("\n【BFS 分层结果】")
    for i, level in enumerate(bfs_with_levels(g, "主楼")):
        print(f"  第{i}层: {level}")

    # ---- 连通性 ----
    print("\n" + "=" * 45)
    print("【连通性】")
    print("全图连通：", is_connected(g))
    print("连通分量：", connected_components(g))

    # 添加孤立节点测试不连通
    g.add_node("孤立楼", 999, 999)
    print("\n添加孤立楼后：")
    print("全图连通：", is_connected(g))
    print("连通分量：", connected_components(g))