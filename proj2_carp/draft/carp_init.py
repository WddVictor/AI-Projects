import sys
import time
import numpy as np
import re
from datetime import datetime
import random
#
instance_file = './samples/egl-s1-A.dat'
termination = 60
random_seed = 0


class Task:
    def __init__(self, path: str):
        self.name = None
        self.vertices = -1
        self.depot = None
        self.num_r_edges = None
        self.num_nr_edges = None
        self.vehicles = None
        self.capacity = None
        self.r_edges_cost = None
        self.graph = None
        self.r_edges = []
        self.nr_edges = []
        self.shortest_path = None
        self.free = None
        self.resolve_file(path)
        # self.neighbor_edge = []
        # self.find_neighbor_edge()
        self.find_free()
        self.find_shortest()

    def find_free(self):
        self.free = []
        for edge in self.r_edges:
            self.free.append((edge.a, edge.b))
            self.free.append((edge.b, edge.a))

    def resolve_file(self, path: str):
        file = open(path)
        while True:
            current_line = file.readline()
            str_array = re.split(r'[\s]', current_line)
            str_array = list(filter(get_rid_of_space, str_array))
            if not str_array:
                continue
            elif str_array[0] == 'NAME':
                self.name = str_array[2]
            elif str_array[0] == 'VERTICES':
                self.vertices = int(str_array[2])
            elif str_array[0] == 'DEPOT':
                self.depot = int(str_array[2]) - 1
            elif str_array[0] == 'REQUIRED':
                self.num_r_edges = int(str_array[3])
            elif str_array[0] == 'NON-REQUIRED':
                self.num_nr_edges = int(str_array[3])
            elif str_array[0] == 'VEHICLES':
                self.vehicles = int(str_array[2])
            elif str_array[0] == 'CAPACITY':
                self.capacity = int(str_array[2])
            elif str_array[0] == 'TOTAL':
                self.r_edges_cost = int(str_array[6])
            elif str_array[0] == 'NODES':
                self.graph = np.empty((self.vertices, self.vertices), dtype=Edge)
                self.shortest_path = np.zeros((self.vertices, self.vertices), dtype=int)
                for i in range(0, self.num_r_edges + self.num_nr_edges):
                    current_line = file.readline()
                    str_array = re.split(r'[\s]', current_line)
                    str_array = np.array(list(filter(get_rid_of_space, str_array))).astype('int')
                    edge = Edge(str_array[0] - 1, str_array[1] - 1, str_array[2], str_array[3])
                    self.graph[(edge.a, edge.b)] = edge
                    self.graph[(edge.b, edge.a)] = edge
                    if edge.is_required_edge():
                        self.r_edges.append(edge)
                    else:
                        self.nr_edges.append(edge)
            if str_array[0] == 'END':
                break
            else:
                continue

    def find_shortest(self):
        short = time.time()
        graph = np.zeros((self.vertices, self.vertices))
        for i in range(0, self.vertices):
            for j in range(0, self.vertices):
                edge = self.graph[i, j]
                if edge:
                    graph[i, j] = edge.cost
                else:
                    if i != j:
                        graph[i, j] = float('inf')
                    else:
                        graph[i, j] = 0
        self.shortest_path = self.floyd(graph.copy())
        # print(self.shortest_path)
        # print('最短路径时间：', time.time() - short)

    def dijkstra(self, graph, src):
        # 判断图是否为空，如果为空直接退出
        result = np.zeros(self.vertices, dtype=int)
        # 判断图是否为空，如果为空直接退出
        if graph is None:
            return None
        nodes = [i for i in range(0, len(graph))]  # 获取图中所有节点
        visited = []  # 表示已经路由到最短路径的节点集合
        if src in nodes:
            visited.append(src)
            nodes.remove(src)
        else:
            return None
        distance = {src: 0}  # 记录源节点到各个节点的距离
        for i in nodes:
            distance[i] = graph[src][i]  # 初始化
        # print(distance)
        path = {src: {src: []}}  # 记录源节点到每个节点的路径
        k = pre = src
        while nodes:
            mid_distance = float('inf')
            for v in visited:
                for d in nodes:
                    new_distance = graph[src][v] + graph[v][d]
                    if new_distance < mid_distance:
                        mid_distance = new_distance
                        graph[src][d] = new_distance  # 进行距离更新
                        k = d
                        pre = v
            distance[k] = mid_distance  # 最短路径
            path[src][k] = [i for i in path[src][pre]]
            path[src][k].append(k)
            # 更新两个节点集合
            visited.append(k)
            nodes.remove(k)
        for point in distance:
            result[point] = distance[point]
        return result

    def floyd(self, graph):
        for k in range(self.vertices):
            for i in range(self.vertices):
                for j in range(self.vertices):
                    if graph[i][j] > graph[i][k] + graph[k][j]:
                        graph[i][j] = graph[i][k] + graph[k][j]
        return graph.astype('int')


class Route:
    def __init__(self, path: list, cost: int, demand: int):
        self.path = path
        self.cost = cost
        self.demand = demand


class Edge:

    def __init__(self, a: int, b: int, cost: int, required: int):
        self.a = a
        self.b = b
        self.cost = cost
        self.required = required

    def is_required_edge(self):
        return self.required

    @staticmethod
    def get_cost(e):
        return e.cost


def get_rid_of_space(e: str):
    return e != ''


def path_scanning(task: Task, rule) -> tuple:
    k = 0
    free = task.free.copy()
    shortest = task.shortest_path
    graph = task.graph
    total = 0
    routes = []
    while free:
        k += 1
        r = []
        load = 0
        cost = 0
        i = task.depot
        d = -1
        while d < float('inf') and free:
            d = float('inf')
            best_edge = None
            for edge_index in free:
                if not graph[edge_index].required + load <= task.capacity:
                    continue
                else:
                    if d == -1 or shortest[i, edge_index[0]] < d:
                        d = shortest[i, edge_index[0]]
                        best_edge = edge_index
                    elif shortest[i, edge_index[0]] == d:
                        if better(edge_index, best_edge, rule, task, load):
                            best_edge = edge_index
            if not best_edge:
                break
            else:
                r.append(best_edge)
                free.remove(best_edge)
                free.remove((best_edge[1], best_edge[0]))
                load += graph[best_edge].required
                part_cost = shortest[i, best_edge[0]] + graph[best_edge].cost
                cost += part_cost
                i = best_edge[1]
        cost += shortest[i, task.depot]
        total += cost
        routes.append(Route(r, cost, load))
    #     print('车辆', k, r, cost)
    # print('总花费：', total)
    return routes, total


def rule1(e1: tuple, e2: tuple, task: Task, load: int):
    shortest = task.shortest_path
    e1_cost = shortest[task.depot, e1[0]]
    e2_cost = shortest[task.depot, e2[0]]
    return e1_cost > e2_cost


def rule2(e1: tuple, e2: tuple, task: Task, load: int):
    shortest = task.shortest_path
    e1_cost = shortest[task.depot, e1[0]]
    e2_cost = shortest[task.depot, e2[0]]
    return e1_cost < e2_cost


def rule3(e1: tuple, e2: tuple, task: Task, load: int):
    graph = task.graph
    edge1, edge2 = graph[e1], graph[e2]
    dem_e1, dem_e2 = edge1.required, edge2.required
    sc_e1, sc_e2 = edge1.cost, edge2.cost
    return dem_e1 / sc_e1 > dem_e2 / sc_e2


def rule4(e1: tuple, e2: tuple, task: Task, load: int):
    graph = task.graph
    edge1, edge2 = graph[e1], graph[e2]
    dem_e1, dem_e2 = edge1.required, edge2.required
    sc_e1, sc_e2 = edge1.cost, edge2.cost
    return dem_e1 / sc_e1 < dem_e2 / sc_e2


def rule5(e1: tuple, e2: tuple, task: Task, load: int):
    shortest = task.shortest_path
    e1_cost = shortest[task.depot, e1[0]]
    e2_cost = shortest[task.depot, e2[0]]
    if load < task.capacity / 2:
        return e1_cost < e2_cost
    else :
        return e1_cost > e2_cost



def better(e1: tuple, e2: tuple, rule, task: Task, load: int) -> bool:
    return rule(e1, e2, task, load)


def resolve_task(path: str) -> str:
    task = Task(path)
    start = datetime.now()
    best = float('inf')
    best_r = None
    rule = [rule1,rule2,rule3,rule4,rule5]
    for i in range(0,2000):
        routes, total_cost = path_scanning(task, rule[random.randint(0,4)])
        if total_cost<best:
            best_r = routes
            best = total_cost
    print(datetime.now()-start)
    result = routes2str_arr(best_r, best)
    return result


def routes2str_arr(routes: list, total_cost: int) -> str:
    s = 's '
    q = 'q '
    for route in routes:
        edges = route.path
        s += '0'
        for edge in edges:
            s += ',(' + str(edge[0] + 1) + ',' + str(edge[1] + 1) + ')'
        s += ',0,'
    s = s[0:-1]
    q += str(total_cost)
    return s + '\n' + q


if __name__ == '__main__':
    # instance_file = sys.argv[1]
    # termination = int(sys.argv[3])
    # random_seed = int(sys.argv[5])
    random.seed(random_seed)
    print(resolve_task(instance_file))
    exit(0)
