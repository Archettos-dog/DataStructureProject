class Graph:
    def __init__(self):
        """
        初始化图。

        内部数据结构：
          self.nodes : dict[str, dict]
              key   = 建筑名称（唯一标识）
              value = 建筑属性，例如 {"x": 120, "y": 340, "alias": "主楼"}

          self.adj : dict[str, dict[str, float]]
              邻接表
              key   = 建筑名称
              value = {邻居名称: 距离(米), ...}

        示例:
          adj = {
              "主楼":  {"教一楼": 150, "图书馆": 200},
              "教一楼": {"主楼":  150, "教二楼": 100},
              ...
          }
        """
        self.nodes: dict[str, dict] = {}
        self.adj:   dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    #  建筑（顶点）操作
    # ------------------------------------------------------------------ #

    def add_node(self, name: str, x: float = 0, y: float = 0, **attrs) -> bool:
        """
        添加一栋建筑（顶点）。

        参数：
            name  : 建筑名称，作为唯一 ID
            x, y  : 在地图上的像素坐标（用于前端绘图）
            attrs : 其他可选属性，如 alias、description 等

        返回：
            True  —— 添加成功
            False —— 建筑已存在，未做修改
        """
        if name in self.nodes:
            return False
        self.nodes[name] = {"x": x, "y": y, **attrs}
        self.adj[name] = {}
        return True

    def delete_node(self, name: str) -> bool:
        """
        删除一栋建筑（顶点）及其所有相关道路（边）。

        参数：
            name : 建筑名称

        返回：
            True  —— 删除成功
            False —— 建筑不存在
        """
        if name not in self.nodes:
            return False
        # 删除所有邻居指向该顶点的边
        for neighbor in self.adj[name]:
            self.adj[neighbor].pop(name, None)
        # 删除顶点自身
        del self.adj[name]
        del self.nodes[name]
        return True

    def update_node(self, name: str, **attrs) -> bool:
        """
        更新建筑属性（坐标或其他字段）。

        参数：
            name  : 建筑名称
            attrs : 要更新的字段，如 x=100, y=200

        返回：
            True  —— 更新成功
            False —— 建筑不存在
        """
        if name not in self.nodes:
            return False
        self.nodes[name].update(attrs)
        return True

    def get_node(self, name: str) -> dict | None:
        """
        获取单个建筑的属性。

        返回：
            属性字典，或 None（不存在时）
        """
        return self.nodes.get(name, None)

    def all_nodes(self) -> list[str]:
        """返回所有建筑名称列表。"""
        return list(self.nodes.keys())

    def node_count(self) -> int:
        """返回建筑总数。"""
        return len(self.nodes)

    # ------------------------------------------------------------------ #
    #  道路（边）操作
    # ------------------------------------------------------------------ #

    def add_edge(self, u: str, v: str, distance: float) -> bool:
        """
        添加一条道路（无向边）。

        参数：
            u, v     : 两端建筑名称
            distance : 道路长度（米），必须 > 0

        返回：
            True  —— 添加成功
            False —— 顶点不存在 / 距离非法 / 边已存在
        """
        if u not in self.nodes or v not in self.nodes:
            return False
        if distance <= 0:
            return False
        if v in self.adj[u]:          # 边已存在，不重复添加
            return False
        self.adj[u][v] = distance
        self.adj[v][u] = distance     # 无向图：双向写入
        return True

    def delete_edge(self, u: str, v: str) -> bool:
        """
        删除一条道路（无向边）。

        参数：
            u, v : 两端建筑名称

        返回：
            True  —— 删除成功
            False —— 边不存在
        """
        if u not in self.adj or v not in self.adj[u]:
            return False
        del self.adj[u][v]
        del self.adj[v][u]
        return True

    def update_edge(self, u: str, v: str, distance: float) -> bool:
        """
        修改道路距离。

        返回：
            True  —— 修改成功
            False —— 边不存在 / 距离非法
        """
        if u not in self.adj or v not in self.adj[u]:
            return False
        if distance <= 0:
            return False
        self.adj[u][v] = distance
        self.adj[v][u] = distance
        return True

    def get_edge(self, u: str, v: str) -> float | None:
        """
        获取两栋楼之间的直接道路距离。

        返回：
            距离（米），或 None（不相邻时）
        """
        return self.adj.get(u, {}).get(v, None)

    def all_edges(self) -> list[tuple[str, str, float]]:
        """
        返回所有道路列表，每条边只返回一次（u < v 字典序）。

        格式：[(建筑A, 建筑B, 距离), ...]
        """
        edges = []
        visited = set()
        for u in self.adj:
            for v, dist in self.adj[u].items():
                key = (min(u, v), max(u, v))
                if key not in visited:
                    visited.add(key)
                    edges.append((u, v, dist))
        return edges

    def edge_count(self) -> int:
        """返回道路总数（每条无向边计 1 次）。"""
        return len(self.all_edges())

    def neighbors(self, name: str) -> dict[str, float]:
        """
        返回某建筑的所有直接相邻建筑及距离。

        返回：
            {邻居名称: 距离, ...}，建筑不存在时返回空字典
        """
        return dict(self.adj.get(name, {}))

    def has_node(self, name: str) -> bool:
        """判断建筑是否存在。"""
        return name in self.nodes

    def has_edge(self, u: str, v: str) -> bool:
        """判断两栋楼之间是否有直接道路。"""
        return u in self.adj and v in self.adj[u]

    # ------------------------------------------------------------------ #
    #  序列化（供 Flask 转 JSON 传给前端）
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """
        将整张图序列化为字典，Flask 可直接 jsonify() 返回给前端。

        格式：
        {
            "nodes": {
                "主楼": {"x": 120, "y": 340},
                ...
            },
            "edges": [
                {"from": "主楼", "to": "教一楼", "distance": 150},
                ...
            ]
        }
        """
        return {
            "nodes": {name: dict(attrs) for name, attrs in self.nodes.items()},
            "edges": [
                {"from": u, "to": v, "distance": dist}
                for u, v, dist in self.all_edges()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Graph":
        """
        从字典（如从 JSON 文件加载）恢复图对象。

        参数：
            data : to_dict() 输出的格式

        返回：
            Graph 实例
        """
        g = cls()
        for name, attrs in data.get("nodes", {}).items():
            x = attrs.pop("x", 0)
            y = attrs.pop("y", 0)
            g.add_node(name, x, y, **attrs)
        for edge in data.get("edges", []):
            g.add_edge(edge["from"], edge["to"], edge["distance"])
        return g

    # ------------------------------------------------------------------ #
    #  调试工具
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"Graph(nodes={self.node_count()}, edges={self.edge_count()})"
        )

    def print_adj(self) -> None:
        """打印邻接表，便于调试。"""
        print(f"{'='*40}")
        print(f"图：{self.node_count()} 个建筑，{self.edge_count()} 条道路")
        print(f"{'='*40}")
        for node, neighbors in self.adj.items():
            neighbor_str = ", ".join(
                f"{nb}({dist}m)" for nb, dist in neighbors.items()
            )
            print(f"  {node:12s} → [{neighbor_str}]")


# ------------------------------------------------------------------ #
#  简单测试（直接运行此文件时执行）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    g = Graph()

    # 添加建筑
    g.add_node("主楼",   x=400, y=500)
    g.add_node("图书馆", x=350, y=300)
    g.add_node("教一楼", x=320, y=460)
    g.add_node("教二楼", x=320, y=600)
    g.add_node("科学会堂", x=530, y=530)

    # 添加道路
    g.add_edge("主楼",   "图书馆",  200)
    g.add_edge("主楼",   "教一楼",  150)
    g.add_edge("主楼",   "科学会堂", 130)
    g.add_edge("教一楼", "教二楼",  120)
    g.add_edge("图书馆", "教一楼",  180)

    g.print_adj()

    print("\n主楼的邻居:", g.neighbors("主楼"))
    print("主楼→图书馆距离:", g.get_edge("主楼", "图书馆"), "米")
    print("序列化:", g.to_dict())

    # 删除建筑
    g.delete_node("科学会堂")
    print("\n删除科学会堂后:")
    g.print_adj()