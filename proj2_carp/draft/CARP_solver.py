import sys
from datetime import datetime
import numpy as np
import re
import random
import multiprocessing as mp

instance_file = './samples/val5A.dat'
termination = 60
random_seed = int(datetime.now().second)


worker_num = 4


class Worker(mp.Process):
    def __init__(self, inQ, outQ, random_seed):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        np.random.seed(random_seed)  # 如果子进程的任务是有随机性的，一定要给每个子进程不同的随机数种子，否则就在重复相同的结果了

    def run(self):
        s = means(2000, 30, 20, 0.2)  # 执行任务
        self.outQ.put(s)  # 返回结果


def create_worker(num):
    '''
    创建子进程备用
    :param num: 多线程数量
    '''
    for i in range(num):
        worker.append(Worker(mp.Queue(), mp.Queue(), np.random.randint(0, 10 ** 9)))
        worker[i].start()


def finish_worker():
    '''
    关闭所有子线程
    '''
    for w in worker:
        w.terminate()


class Problem:
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
        return

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


class Solution:
    def __init__(self, routes: list, total_cost: int, total_load: int):
        self.routes = routes
        self.total_cost = total_cost
        self.total_load = total_load
        self.total_violation = None

    @staticmethod
    def get_total_cost(solution):
        return solution.total_cost


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


def path_scanning(free, routes) -> tuple:
    k = 0
    shortest = problem.shortest_path
    graph = problem.graph
    total_cost = 0
    total_load = 0
    while free:
        k += 1
        r = []
        load = 0
        cost = 0
        i = problem.depot
        d = -1
        while free:
            d = float('inf')
            temp_edges = []
            best_edges = []
            for edge_index in free:
                if shortest[i, edge_index[0]] < d:
                    temp_edges = [edge_index]
                    d = shortest[i, edge_index[0]]
                elif shortest[i, edge_index[0]] == d:
                    temp_edges.append(edge_index)
            for temp_edge in temp_edges:
                if load + graph[temp_edge].required <= problem.capacity:
                    best_edges.append(temp_edge)
            if not best_edges:
                break
            else:
                best_edge = best_edges[random.randint(0, len(best_edges) - 1)]
                r.append(best_edge)
                free.remove(best_edge)
                free.remove((best_edge[1], best_edge[0]))
                load += graph[best_edge].required
                part_cost = shortest[i, best_edge[0]] + graph[best_edge].cost
                cost += part_cost
                i = best_edge[1]
        cost += shortest[i, problem.depot]
        total_cost += cost
        total_load += load
        routes.append(Route(r, cost, load))
    #     print('车辆', k, r, cost)
    # print('总花费：', total_cost)
    # print('总载量：', total_load)
    return routes, total_cost, total_load


def generate_init():
    routes, total_cost, total_load = path_scanning(problem.free.copy(), [])
    return Solution(routes, total_cost, total_load)


def crossover(s1: Solution, s2: Solution) -> Solution:
    shortest = problem.shortest_path
    routes1 = s1.routes
    routes2 = s2.routes
    r1_random, r2_random = random.randint(0, len(routes1) - 1), random.randint(0, len(routes2) - 1)
    r1, r2 = routes1[r1_random], routes2[r2_random]
    r1_path, r2_path = r1.path.copy(), r2.path.copy()
    r11_random, r21_random = random.randint(0, len(r1_path)), random.randint(0, len(r2_path))
    r11, r12 = r1_path[0:r11_random], r1_path[r11_random:]
    r21, r22 = r2_path[0:r21_random], r2_path[r21_random:]
    lack = set(r12) - set(r22)
    duplicate = set(r22) - set(r12)
    r22, cost_r22, load_r22 = delete_duplicate(r22, duplicate)
    offspring_path = r11 + r22
    if offspring_path:
        offspring_cost = find_parttern_cost(offspring_path) + shortest[problem.depot, offspring_path[0][0]] + \
                         shortest[offspring_path[-1][1], problem.depot]
    else:
        offspring_cost = 0
    offspring = Route(offspring_path, offspring_cost,
                      find_parttern_load(r11) + load_r22)
    sx = sbx(lack, offspring, routes1.copy(), s1.total_cost - r1.cost + offspring.cost,
             s1.total_load - r1.demand + offspring.demand, r1_random)
    return sx


def delete_duplicate(parttern: list, duplicate: set, ) -> tuple:
    graph = problem.graph
    shortest = problem.shortest_path
    parttern_cost = find_parttern_cost(parttern)
    parttern_load = find_parttern_load(parttern)
    if len(parttern) == 1 and duplicate:
        duplicate.pop()
        parttern.pop()
        return [], 0, 0
    while duplicate:
        edge = duplicate.pop()
        index = parttern.index(edge)
        current_edge = graph[edge]
        if index == 0:
            next_edge = parttern[1]
            parttern_cost = parttern_cost - current_edge.cost - shortest[(edge[1], next_edge[0])]
        elif index == len(parttern) - 1:
            pre_edge = parttern[index - 1]
            parttern_cost = parttern_cost - current_edge.cost - shortest[(pre_edge[1], edge[0])]
        else:
            pre_edge = parttern[index - 1]
            next_edge = parttern[index + 1]
            parttern_cost = parttern_cost - current_edge.cost - shortest[(pre_edge[1], edge[0])] - \
                            shortest[(edge[1], next_edge[0])] + shortest[pre_edge[1], next_edge[0]]
        parttern_load = parttern_load - current_edge.required
        parttern.pop(index)
        if len(parttern) == 1:
            if duplicate:
                duplicate.pop()
                parttern.pop()
                return [], 0, 0
    return parttern, parttern_cost, parttern_load


def find_best_position(edges: list, route: Route) -> tuple:
    depot = problem.depot
    shortest = problem.shortest_path
    best_cost = float('inf')
    best_position = None
    best_reverse = False
    best_load = -1
    path = route.path
    path_cost = route.cost
    path_load = route.demand
    head, tail = edges[0][0], edges[-1][1]
    edge_cost = find_parttern_cost(edges)
    edge_load = find_parttern_load(edges)
    if route.cost == 0:
        origin = shortest[depot, head] + edge_cost + shortest[tail, depot]
        head, tail = tail, head
        after = shortest[depot, head] + edge_cost + shortest[tail, depot]
        if after < origin:
            return 0, after, edge_load, True
        else:
            return 0, origin, edge_load, False
    for counter in range(0, 2):
        if counter == 1:
            reverse = True
            tail, head = head, tail
        else:
            reverse = False
        for i in range(0, len(path) + 1):
            pre_index, next_index = i - 1, i
            pre_edge, next_edge = (depot, depot), (depot, depot)
            temp_path_cost = path_cost
            temp_path_load = path_load
            if pre_index != -1:
                pre_edge = path[pre_index]
            if next_index != len(path):
                next_edge = path[next_index]
            temp_path_cost = temp_path_cost - shortest[pre_edge[1], next_edge[0]] + edge_cost + shortest[
                pre_edge[1], head] + shortest[tail, next_edge[0]]
            temp_path_load += edge_load
            if temp_path_cost < best_cost:
                best_position = i
                best_cost = temp_path_cost
                best_load = temp_path_load
                best_reverse = reverse
    return best_position, best_cost, best_load, best_reverse


def sbx(lack: set, offspring: Route, routes: list, total_cost: int, total_load: int,
        r1_index: int) -> Solution:
    capacity = problem.capacity
    graph = problem.graph
    for edge in lack:
        best_index = None
        best_route = None
        best_cost = float('inf')
        best_reverse = False
        best_load = -1
        new_route_index = None
        for i in range(0, len(routes)):
            if i == r1_index:
                route = offspring
            else:
                route = routes[i]
            route_load = route.demand
            if route_load + graph[edge].required > capacity:
                continue
            else:
                temp_index, temp_cost, temp_load, temp_reverse = find_best_position([edge], route)
                if temp_cost < best_cost:
                    best_index, best_cost, best_load = temp_index, temp_cost, temp_load
                    best_route = route
                    best_reverse = temp_reverse
                    new_route_index = i
        total_cost = total_cost - best_route.cost + best_cost
        total_load = total_load - best_route.demand + best_load
        if best_reverse:
            edge = (edge[1], edge[0])
        if best_route != offspring:
            best_path = best_route.path.copy()
            best_path.insert(best_index, edge)
            new_route = Route(best_path, best_cost, best_load)
            routes[new_route_index] = new_route
        else:
            best_route.path.insert(best_index, edge)
            best_route.cost, best_route.demand = best_cost, best_load
    routes[r1_index] = offspring
    return Solution(routes, total_cost, total_load)


def find_parttern_cost(r_parttern: list) -> int:
    graph = problem.graph
    shortest = problem.shortest_path
    if not r_parttern:
        return 0
    edge = r_parttern[0]
    parttern_cost = 0

    for index in range(1, len(r_parttern)):
        dest = edge[1]
        next_edge = r_parttern[index]
        parttern_cost += graph[edge].cost + shortest[(dest, next_edge[0])]
        edge = next_edge
    parttern_cost += graph[edge].cost
    return parttern_cost


def find_parttern_load(r_parttern: list) -> int:
    graph = problem.graph
    parttern_load = 0
    for edge in r_parttern:
        parttern_load += graph[edge].required
    return parttern_load


def single_insertion(sx: Solution) -> Solution:
    capacity = problem.capacity
    graph = problem.graph
    depot = problem.depot
    shortest = problem.shortest_path
    routes = sx.routes.copy()
    best_cost = float('inf')
    best_load = 0
    best_origin_cost = 0
    best_origin_load = 0
    best_i, best_j = None, None
    total_cost = sx.total_cost
    total_load = sx.total_load
    best_i_route, best_j_route = None, None
    best_reverse = False
    for i_route in range(0, len(routes)):
        route = routes[i_route]
        path = route.path
        for i in range(0, len(path)):
            edge = path[i]
            pre_index, next_index = i - 1, i + 1
            pre_edge, next_edge = (depot, depot), (depot, depot)
            if pre_index != -1:
                pre_edge = path[pre_index]
            if next_index != len(path):
                next_edge = path[next_index]
            route.path.pop(i)
            origin_cost = route.cost
            origin_load = route.demand
            removed_cost = graph[edge].cost + shortest[pre_edge[1], edge[0]] + shortest[
                edge[1], next_edge[0]] - shortest[pre_edge[1], next_edge[0]]
            removed_load = graph[edge].required
            route.cost = origin_cost - removed_cost
            route.demand = origin_load - removed_load
            for j_route in range(0, len(routes)):
                candidate_route = routes[j_route]
                posit, t_cost, t_load, t_reverse = find_best_position([edge], candidate_route)
                if t_cost < best_cost and t_load <= capacity:
                    best_reverse = t_reverse
                    best_cost, best_load = t_cost, t_load
                    best_origin_cost, best_origin_load = route.cost, route.demand
                    best_i, best_j = i, posit
                    best_i_route, best_j_route = i_route, j_route
            route.path.insert(i, edge)
            route.cost = origin_cost
            route.demand = origin_load
    if best_i != None:
        i_route = routes[best_i_route]
        j_route = routes[best_j_route]
        removed = i_route.path[best_i]
        if best_reverse:
            removed = (removed[1], removed[0])
        new_i_path, new_j_path = i_route.path.copy(), j_route.path.copy()
        if new_i_path == new_j_path:
            new_i_path = new_j_path
        new_i_path.pop(best_i)
        new_j_path.insert(best_j, removed)
        total_cost = sx.total_cost - i_route.cost + best_cost
        if best_j_route != best_i_route:
            total_cost += best_origin_cost - j_route.cost
        total_load = sx.total_load - i_route.demand - j_route.demand + best_load + best_origin_load
        if best_i_route == best_j_route:
            routes[best_i_route] = Route(new_i_path, best_origin_cost + best_cost - routes[best_i_route].cost,
                                         routes[best_i_route].demand)
        else:
            routes[best_i_route] = Route(new_i_path, best_origin_cost, best_origin_load)
            routes[best_j_route] = Route(new_j_path, best_cost, best_load)
    s = Solution(routes, total_cost, total_load)
    return s


def double_insertion(sx: Solution) -> Solution:
    capacity = problem.capacity
    graph = problem.graph
    depot = problem.depot
    shortest = problem.shortest_path
    routes = sx.routes.copy()
    best_cost = float('inf')
    best_load = 0
    best_origin_cost = 0
    best_origin_load = 0
    best_i, best_j = None, None
    total_cost = sx.total_cost
    total_load = sx.total_load
    best_i_route, best_j_route = None, None
    best_reverse = False
    for i_route in range(0, len(routes)):
        route = routes[i_route]
        path = route.path
        if len(path) > 1:
            for i in range(0, len(path) - 1):
                edge = path[i:i + 2]
                pre_index, next_index = i - 1, i + 2
                pre_edge, next_edge = (depot, depot), (depot, depot)
                if pre_index != -1:
                    pre_edge = path[pre_index]
                if next_index != len(path):
                    next_edge = path[next_index]
                route.path.pop(i)
                route.path.pop(i)
                origin_cost = route.cost
                origin_load = route.demand
                removed_cost = find_parttern_cost(edge) + shortest[pre_edge[1], edge[0][0]] + shortest[
                    edge[1][1], next_edge[0]] - shortest[pre_edge[1], next_edge[0]]

                removed_load = graph[edge[0]].required + graph[edge[1]].required
                route.cost = origin_cost - removed_cost
                route.demand = origin_load - removed_load
                for j_route in range(0, len(routes)):
                    candidate_route = routes[j_route]
                    posit, t_cost, t_load, t_reverse = find_best_position(edge, candidate_route)

                    if t_cost < best_cost and t_load <= capacity:
                        best_cost, best_load = t_cost, t_load
                        best_origin_cost, best_origin_load = route.cost, route.demand
                        best_i, best_j = i, posit
                        best_i_route, best_j_route = i_route, j_route
                        best_reverse = t_reverse
                route.path.insert(i, edge[0])
                route.path.insert(i + 1, edge[1])
                route.cost = origin_cost
                route.demand = origin_load
    if best_i != None:
        i_route = routes[best_i_route]
        j_route = routes[best_j_route]
        removed = i_route.path[best_i:best_i + 2]
        if best_reverse:
            removed = [(removed[1][1], removed[1][0]), (removed[0][1], removed[0][0])]
        new_i_path, new_j_path = i_route.path.copy(), j_route.path.copy()
        if new_i_path == new_j_path:
            new_i_path = new_j_path
        new_i_path.pop(best_i)
        new_i_path.pop(best_i)
        new_j_path.insert(best_j, removed[0])
        new_j_path.insert(best_j + 1, removed[1])
        total_cost = sx.total_cost - i_route.cost + best_cost
        if best_j_route != best_i_route:
            total_cost += best_origin_cost - j_route.cost
        total_load = sx.total_load - i_route.demand - j_route.demand + best_load + best_origin_load
        if best_i_route == best_j_route:
            routes[best_i_route] = Route(new_i_path, best_origin_cost + best_cost - routes[best_i_route].cost,
                                         routes[best_i_route].demand)
        else:
            routes[best_i_route] = Route(new_i_path, best_origin_cost, best_origin_load)
            routes[best_j_route] = Route(new_j_path, best_cost, best_load)

    s = Solution(routes, total_cost, total_load)
    return s


def swap(sx: Solution) -> Solution:
    depot, capacity, graph, shortest = problem.depot, problem.capacity, problem.graph, problem.shortest_path
    routes = sx.routes.copy()
    best_route_i_index, best_route_j_index = None, None
    best_edge_i_index, best_edge_j_index = None, None
    best_route_i_cost, best_route_i_load = float('inf'), None
    best_route_j_cost, best_route_j_load = float('inf'), None
    best_total_cost = float('inf')
    best_i_reverse, best_j_reverse = False, False
    for counter_i in range(0, 2):
        if counter_i == 1:
            reverse_i = True
        else:
            reverse_i = False
        for route_i_index in range(0, len(routes)):
            route_i = routes[route_i_index]
            route_i_path = route_i.path
            for edge_i_index in range(0, len(route_i_path)):
                edge_i = route_i_path[edge_i_index]
                pre_edge_i, next_edge_i = (depot, depot), (depot, depot)
                if edge_i_index - 1 != -1:
                    pre_edge_i = route_i_path[edge_i_index - 1]
                if edge_i_index + 1 != len(route_i_path):
                    next_edge_i = route_i_path[edge_i_index + 1]
                if reverse_i:
                    edge_i = (edge_i[1], edge_i[0])
                for counter_j in range(0, 2):
                    if counter_j == 1:
                        reverse_j = True
                    else:
                        reverse_j = False
                    for route_j_index in range(route_i_index, len(routes)):
                        route_j = routes[route_j_index]
                        route_j_path = route_j.path
                        if route_i_index == route_j_index:
                            left = edge_i_index
                        else:
                            left = 0
                        for edge_j_index in range(left, len(route_j_path)):
                            edge_j = route_j_path[edge_j_index]
                            pre_edge_j, next_edge_j = (depot, depot), (depot, depot)
                            if edge_j_index - 1 != -1:
                                pre_edge_j = route_j_path[edge_j_index - 1]
                            if edge_j_index + 1 != len(route_j_path):
                                next_edge_j = route_j_path[edge_j_index + 1]
                            if reverse_j:
                                edge_j = (edge_j[1], edge_j[0])
                            swap_edge_i_cost = graph[edge_i].cost + shortest[
                                pre_edge_i[1], route_i_path[edge_i_index][0]] + shortest[
                                                   route_i_path[edge_i_index][1], next_edge_i[0]] - graph[edge_j].cost - \
                                               shortest[pre_edge_i[1], edge_j[0]] - \
                                               shortest[edge_j[1], next_edge_i[0]]
                            swap_edge_j_cost = graph[edge_j].cost + shortest[
                                pre_edge_j[1], route_j_path[edge_j_index][0]] + shortest[
                                                   route_j_path[edge_j_index][1], next_edge_j[0]] - graph[edge_i].cost - \
                                               shortest[pre_edge_j[1], edge_i[0]] - \
                                               shortest[edge_i[1], next_edge_j[0]]
                            if route_i_index != route_j_index:
                                temp_route_i_cost, temp_route_j_cost = route_i.cost - swap_edge_i_cost, route_j.cost - swap_edge_j_cost

                                temp_route_i_load, temp_route_j_load = route_i.demand - graph[edge_i].required + graph[
                                    edge_j].required, route_j.demand - graph[edge_j].required + graph[edge_i].required
                                temp_total_cost = sx.total_cost - route_i.cost - route_j.cost + temp_route_i_cost + temp_route_j_cost
                            else:
                                if edge_i_index == edge_j_index - 1:
                                    temp_route_i_cost = temp_route_j_cost = route_i.cost - shortest[
                                        pre_edge_i[1], route_i_path[edge_i_index][0]] - shortest[
                                                                                route_j_path[edge_j_index][1],
                                                                                next_edge_j[0]] - shortest[
                                                                                route_i_path[edge_i_index][1],
                                                                                route_j_path[edge_j_index][0]] + \
                                                                            shortest[
                                                                                pre_edge_i[1], edge_j[0]] + shortest[
                                                                                edge_j[1], edge_i[0]] + shortest[
                                                                                edge_i[1], next_edge_j[0]]
                                elif edge_i_index == edge_j_index:
                                    temp_route_i_cost = temp_route_j_cost = route_i.cost - shortest[
                                        pre_edge_i[1], route_i_path[edge_i_index][0]] - shortest[
                                                                                route_i_path[edge_i_index][1],
                                                                                next_edge_i[0]] + \
                                                                            shortest[pre_edge_i[1], edge_i[0]] + \
                                                                            shortest[edge_i[1], next_edge_i[0]]
                                else:
                                    temp_route_i_cost = temp_route_j_cost = route_i.cost - swap_edge_i_cost - swap_edge_j_cost
                                temp_total_cost = sx.total_cost - route_i.cost + temp_route_i_cost

                                temp_route_i_load = temp_route_j_load = route_i.demand

                            if temp_total_cost < best_total_cost and temp_route_j_load < capacity and temp_route_i_load < capacity:
                                best_total_cost = temp_total_cost
                                best_route_i_index, best_route_j_index = route_i_index, route_j_index
                                best_edge_i_index, best_edge_j_index = edge_i_index, edge_j_index
                                best_route_i_cost, best_route_j_cost = temp_route_i_cost, temp_route_j_cost
                                best_route_i_load, best_route_j_load = temp_route_i_load, temp_route_j_load
                                best_i_reverse, best_j_reverse = reverse_i, reverse_j
    if best_route_i_index != None:
        route_i, route_j = routes[best_route_i_index], routes[best_route_j_index]
        route_i_path, route_j_path = route_i.path.copy(), route_j.path.copy()
        edge_i, edge_j = route_i_path[best_edge_i_index], route_j_path[best_edge_j_index]
        route_i_cost, route_j_cost = best_route_i_cost, best_route_j_cost
        route_i_load, route_j_load = best_route_i_load, best_route_j_load
        if best_i_reverse:
            edge_i = (edge_i[1], edge_i[0])
        if best_j_reverse:
            edge_j = (edge_j[1], edge_j[0])
        if best_route_i_index == best_route_j_index:
            route_i_path[best_edge_i_index], route_i_path[best_edge_j_index] = edge_j, edge_i
            route_i = Route(route_i_path, route_i_cost, route_i_load)
            routes[best_route_i_index] = route_i
        else:
            route_i_path[best_edge_i_index], route_j_path[best_edge_j_index] = edge_j, edge_i
            route_i = Route(route_i_path, route_i_cost, route_i_load)
            route_j = Route(route_j_path, route_j_cost, route_j_load)
            routes[best_route_i_index] = route_i
            routes[best_route_j_index] = route_j

    return Solution(routes, best_total_cost, sx.total_load)


def local_search(sx: Solution) -> Solution:
    solutions = [single_insertion(sx), double_insertion(sx), swap(sx)]
    solutions.sort(key=Solution.get_total_cost)
    return solutions[0]


def means(psize: int, opsize: int, ubtrial: int, pls: float) -> Solution:
    pop = []
    pop_set = set()
    while len(pop) < psize:
        ntrial = 0
        s_init = None
        while ntrial != ubtrial:
            ntrial += 1
            s_init = generate_init()
            if not pop_set.__contains__(s_init.total_cost):
                break
        if pop_set.__contains__(s_init.total_cost):
            break
        pop.append(s_init)
        pop_set.add(s_init.total_cost)
    psize = len(pop)
    pop.sort(key=Solution.get_total_cost)
    while True:
        popt = pop.copy()
        for i in range(0, opsize):
            s1, s2 = pop[random.randint(0, len(pop) - 1)], pop[random.randint(0, len(pop) - 1)]
            sx = crossover(s1, s2)
            r = random.random()
            if r < pls:
                sls = local_search(sx)
                if not pop_set.__contains__(sls):
                    popt.append(sls)
                    pop_set.add(sls.total_cost)
                elif not pop_set.__contains__(sx):
                    popt.append(sx)
                    pop_set.add(sx.total_cost)
            elif not pop_set.__contains__(sx):
                popt.append(sx)
                pop_set.add(sx.total_cost)
        popt.sort(key=Solution.get_total_cost)
        pop = popt[0:psize]
        if (datetime.now() - start).seconds > termination - 4:
            return pop[0]


def resolve_problem() -> str:
    solution = []
    create_worker(worker_num)
    for i in range(0, worker_num):
        solution.append(worker[i].outQ.get())
    finish_worker()
    solution.sort(key=Solution.get_total_cost)
    result = solution2format(solution[0])
    return result


def find_route_cost(path: list) -> int:
    pre, next = (0, 0), (0, 0)
    graph = problem.graph
    shortest = problem.shortest_path
    t_cost = 0
    for edge in path:
        t_cost += shortest[pre[1], edge[0]] + graph[edge].cost
        pre = edge
    t_cost += shortest[pre[1], next[0]]
    return t_cost


def find_total_cost(solution: Solution) -> int:
    cost = 0
    for route in solution.routes:
        cost += find_route_cost(route.path)
    return cost


def find_total_load(solution: Solution) -> int:
    load = 0
    for route in solution.routes:
        load += find_parttern_load(route.path)
    return load


def solution2format(solution: Solution) -> str:
    s = 's '
    q = 'q '
    routes, total_cost = solution.routes, solution.total_cost
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
    start = datetime.now()
    random.seed(random_seed)
    worker = []
    problem = Problem(instance_file)
    print(resolve_problem())
    exit(0)
