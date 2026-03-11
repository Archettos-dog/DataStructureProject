"""
app.py —— Flask 后端服务
提供 REST API，连接前端操作与后端图算法。

启动方式：
    python app.py
服务地址：
    http://127.0.0.1:5000

API 一览：
    GET  /api/graph                          获取完整图数据（建筑+道路）
    GET  /api/shortest_path?from=A&to=B      查询最短路径
    GET  /api/traverse?start=A&method=dfs    遍历校园（dfs / bfs）
    POST /api/node                           添加建筑
    DELETE /api/node/<name>                  删除建筑
    POST /api/edge                           添加道路
    DELETE /api/edge/<u>/<v>                 删除道路
    GET  /api/connected                      检查全图连通性
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
from flask import Flask, request, jsonify, render_template
from data import build_graph
from algorithms import dijkstra, dfs, bfs, is_connected, connected_components

app = Flask(__name__)

# ------------------------------------------------------------------ #
#  全局图对象（服务启动时从 data.py 初始化，运行期间保持在内存中）
# ------------------------------------------------------------------ #
graph = build_graph()


# ------------------------------------------------------------------ #
#  工具函数
# ------------------------------------------------------------------ #

def ok(data: dict | list = None, **kwargs):
    """返回成功响应。"""
    payload = {"status": "ok"}
    if data is not None:
        payload["data"] = data
    payload.update(kwargs)
    return jsonify(payload)


def err(message: str, code: int = 400):
    """返回错误响应。"""
    return jsonify({"status": "error", "message": message}), code


# ------------------------------------------------------------------ #
#  页面路由
# ------------------------------------------------------------------ #

@app.route("/")
def index():
    """返回前端页面 index.html。"""
    return render_template("index.html")


# ------------------------------------------------------------------ #
#  API：图数据
# ------------------------------------------------------------------ #

@app.route("/api/graph", methods=["GET"])
def get_graph():
    """
    获取完整图数据，前端用此初始化地图。

    响应示例：
    {
        "status": "ok",
        "data": {
            "nodes": {"主楼": {"x": 620, "y": 560, "category": "teaching"}, ...},
            "edges": [{"from": "主楼", "to": "教一楼", "distance": 130}, ...]
        }
    }
    """
    return ok(graph.to_dict())


# ------------------------------------------------------------------ #
#  API：最短路径
# ------------------------------------------------------------------ #

@app.route("/api/shortest_path", methods=["GET"])
def shortest_path():
    """
    查询两栋建筑之间的最短路径。

    查询参数：
        from : 起点建筑名称
        to   : 终点建筑名称

    响应示例（成功）：
    {
        "status": "ok",
        "data": {
            "path": ["学八公寓", "学五学八路口", "学三学四路口", "学四公寓"],
            "distance": 170,
            "description": "从【学八公寓】到【学四公寓】..."
        }
    }

    响应示例（不连通）：
    {
        "status": "error",
        "message": "【A】与【B】之间不存在可达路径"
    }
    """
    start = request.args.get("from", "").strip()
    end   = request.args.get("to",   "").strip()

    if not start or not end:
        return err("缺少参数：需要 from 和 to")
    if not graph.has_node(start):
        return err(f"建筑不存在：{start}")
    if not graph.has_node(end):
        return err(f"建筑不存在：{end}")

    result = dijkstra(graph, start, end)
    if result is None:
        return err(f"【{start}】与【{end}】之间不存在可达路径")

    path, distance = result
    steps = []
    for i in range(len(path) - 1):
        seg_dist = graph.get_edge(path[i], path[i + 1])
        steps.append({
            "from": path[i],
            "to":   path[i + 1],
            "distance": seg_dist
        })

    description = (
        f"从【{start}】到【{end}】：\n" +
        " → ".join(path) +
        f"\n总距离 {distance:.0f} 米，共 {len(path)-1} 段"
    )

    return ok({
        "path":        path,
        "distance":    distance,
        "steps":       steps,       # 每段的起止点与距离，供前端逐段高亮
        "description": description
    })


# ------------------------------------------------------------------ #
#  API：遍历
# ------------------------------------------------------------------ #

@app.route("/api/traverse", methods=["GET"])
def traverse():
    """
    从指定建筑出发遍历所有可达建筑。

    查询参数：
        start  : 起点建筑名称
        method : 遍历方式，dfs 或 bfs（默认 bfs）

    响应示例：
    {
        "status": "ok",
        "data": {
            "method": "bfs",
            "start": "主楼",
            "order": ["主楼", "行政办公楼", "教一楼", ...],
            "count": 41
        }
    }
    """
    start  = request.args.get("start",  "").strip()
    method = request.args.get("method", "bfs").strip().lower()

    if not start:
        return err("缺少参数：需要 start")
    if not graph.has_node(start):
        return err(f"建筑不存在：{start}")
    if method not in ("dfs", "bfs"):
        return err("method 只能是 dfs 或 bfs")

    order = dfs(graph, start) if method == "dfs" else bfs(graph, start)

    return ok({
        "method": method,
        "start":  start,
        "order":  order,
        "count":  len(order)
    })


# ------------------------------------------------------------------ #
#  API：建筑（顶点）增删
# ------------------------------------------------------------------ #

@app.route("/api/node", methods=["POST"])
def add_node():
    """
    添加一栋新建筑。

    请求体（JSON）：
    {
        "name":     "新楼",
        "x":        400,
        "y":        300,
        "category": "teaching"   // 可选，默认 "other"
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return err("请求体必须是 JSON")

    name     = str(body.get("name", "")).strip()
    x        = body.get("x", 0)
    y        = body.get("y", 0)
    category = str(body.get("category", "other")).strip()

    if not name:
        return err("缺少字段：name")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return err("x 和 y 必须是数字")

    success = graph.add_node(name, x=x, y=y, category=category)
    if not success:
        return err(f"建筑已存在：{name}")

    return ok({"name": name, "x": x, "y": y, "category": category},
              message=f"建筑【{name}】添加成功")


@app.route("/api/node/<path:name>", methods=["DELETE"])
def delete_node(name: str):
    """
    删除一栋建筑及其所有相关道路。

    URL 参数：
        name : 建筑名称（支持中文，Flask 的 path 转换器允许斜线外的任意字符）
    """
    name = name.strip()
    if not graph.has_node(name):
        return err(f"建筑不存在：{name}", 404)

    # 记录删除前的邻居数，用于提示
    neighbor_count = len(graph.neighbors(name))
    graph.delete_node(name)

    return ok(message=f"建筑【{name}】及其 {neighbor_count} 条相关道路已删除")


# ------------------------------------------------------------------ #
#  API：道路（边）增删
# ------------------------------------------------------------------ #

@app.route("/api/edge", methods=["POST"])
def add_edge():
    """
    添加一条道路。

    请求体（JSON）：
    {
        "from":     "主楼",
        "to":       "科学会堂",
        "distance": 230
    }
    """
    body = request.get_json(silent=True)
    if not body:
        return err("请求体必须是 JSON")

    u        = str(body.get("from", "")).strip()
    v        = str(body.get("to",   "")).strip()
    distance = body.get("distance")

    if not u or not v:
        return err("缺少字段：from 或 to")
    if distance is None:
        return err("缺少字段：distance")
    if not isinstance(distance, (int, float)) or distance <= 0:
        return err("distance 必须是正数")
    if not graph.has_node(u):
        return err(f"建筑不存在：{u}")
    if not graph.has_node(v):
        return err(f"建筑不存在：{v}")

    success = graph.add_edge(u, v, distance)
    if not success:
        return err(f"道路已存在：{u} ↔ {v}")

    return ok({"from": u, "to": v, "distance": distance},
              message=f"道路【{u} ↔ {v}】添加成功，距离 {distance} 米")


@app.route("/api/edge/<path:u_v>", methods=["DELETE"])
def delete_edge(u_v: str):
    """
    删除一条道路。

    URL 格式：/api/edge/<建筑A>/<建筑B>
    例：DELETE /api/edge/主楼/科学会堂

    注意：因建筑名含中文，前端需对名称做 encodeURIComponent 编码。
    """
    parts = u_v.split("/")
    if len(parts) != 2:
        return err("URL 格式错误，应为 /api/edge/<建筑A>/<建筑B>")

    u, v = parts[0].strip(), parts[1].strip()

    if not graph.has_edge(u, v):
        return err(f"道路不存在：{u} ↔ {v}", 404)

    graph.delete_edge(u, v)
    return ok(message=f"道路【{u} ↔ {v}】已删除")


# ------------------------------------------------------------------ #
#  API：连通性
# ------------------------------------------------------------------ #

@app.route("/api/connected", methods=["GET"])
def check_connected():
    """
    检查全图连通性，并列出所有连通分量。

    响应示例（连通）：
    {
        "status": "ok",
        "data": {
            "is_connected": true,
            "component_count": 1,
            "components": [["主楼", "教一楼", ...]]
        }
    }
    """
    components = connected_components(graph)
    return ok({
        "is_connected":    is_connected(graph),
        "component_count": len(components),
        "components":      components
    })


# ------------------------------------------------------------------ #
#  启动
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    print("=" * 50)
    print("  北邮校园导航系统后端已启动")
    print("  访问地址：http://127.0.0.1:5000")
    print("=" * 50)
    app.run(debug=True)