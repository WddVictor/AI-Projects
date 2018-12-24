import sys
import time
import numpy as np
import re
import random
#
# instance_file = './samples/gdb10.dat'
# best_c = 275
# termination = 60
# random_seed = int(time.time())
# problem = None


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

    # def find_neighbor_edge(self):
    #     for i in range(0, self.vertices):
    #         self.neighbor_edge.append([])
    #     for edge in self.r_edges:
    #         a = edge.a
    #         b = edge.b
    #         self.neighbor_edge[a].append(edge)
    #         self.neighbor_edge[b].append(edge)

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


def path_scanning(rule) -> Solution:
    k = 0
    free = problem.free.copy()
    shortest = problem.shortest_path
    graph = problem.graph
    total_cost = 0
    total_load = 0
    routes = []
    while free:
        k += 1
        r = []
        load = 0
        cost = 0
        i = problem.depot
        d = -1
        while d < float('inf') and free:
            d = float('inf')
            best_edge = None
            for edge_index in free:
                if not graph[edge_index].required + load <= problem.capacity:
                    continue
                else:
                    if d == -1 or shortest[i, edge_index[0]] < d:
                        d = shortest[i, edge_index[0]]
                        best_edge = edge_index
                    elif shortest[i, edge_index[0]] == d:
                        if better(edge_index, best_edge, rule, load):
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
        cost += shortest[i, problem.depot]
        total_cost += cost
        total_load += load
        routes.append(Route(r, cost, load))
    #     print('车辆', k, r, cost)
    # print('总花费：', total_cost)
    # print('总载量：', total_load)
    solution = Solution(routes, total_cost, total_load)
    return solution


def rule1(e1: tuple, e2: tuple, load: int):
    shortest = problem.shortest_path
    e1_cost = shortest[problem.depot, e1[0]]
    e2_cost = shortest[problem.depot, e2[0]]
    if e1_cost > e2_cost:
        return True
    elif e1_cost < e2_cost:
        return False
    else:
        if random.random() < 0.5:
            return True
        else:
            return False


def rule2(e1: tuple, e2: tuple, load: int):
    shortest = problem.shortest_path
    e1_cost = shortest[problem.depot, e1[0]]
    e2_cost = shortest[problem.depot, e2[0]]
    if e1_cost < e2_cost:
        return True
    elif e1_cost > e2_cost:
        return False
    else:
        if random.random() < 0.5:
            return True
        else:
            return False


def rule3(e1: tuple, e2: tuple, load: int):
    graph = problem.graph
    edge1, edge2 = graph[e1], graph[e2]
    dem_e1, dem_e2 = edge1.required, edge2.required
    sc_e1, sc_e2 = edge1.cost, edge2.cost
    term1 = dem_e1 / sc_e1
    term2 = dem_e2 / sc_e2
    if term1 > term2:
        return True
    elif term1 < term2:
        return False
    else:
        if random.random() < 0.5:
            return True
        else:
            return False


def rule4(e1: tuple, e2: tuple, load: int):
    graph = problem.graph
    edge1, edge2 = graph[e1], graph[e2]
    dem_e1, dem_e2 = edge1.required, edge2.required
    sc_e1, sc_e2 = edge1.cost, edge2.cost
    term1 = dem_e1 / sc_e1
    term2 = dem_e2 / sc_e2
    if term1 < term2:
        return True
    elif term1 > term2:
        return False
    else:
        if random.random() < 0.5:
            return True
        else:
            return False


def rule5(e1: tuple, e2: tuple, load: int):
    shortest = problem.shortest_path
    e1_cost = shortest[problem.depot, e1[0]]
    e2_cost = shortest[problem.depot, e2[0]]
    if load < problem.capacity / 2:
        return e1_cost < e2_cost
    elif load > problem.capacity / 2:
        return e1_cost > e2_cost
    else:
        if random.random() < 0.5:
            return True
        else:
            return False


def better(e1: tuple, e2: tuple, rule, load: int) -> bool:
    return rule(e1, e2, load)


def generate_init():
    r = int(random.random() * 5)
    rules = [rule1, rule2, rule3, rule4, rule5]
    return path_scanning(rules[r])


def stochastic_ranking(popt: list) -> list:
    for i in range(0, len(popt) - 1):
        for j in range(0, len(popt) - i - 1):
            if popt[j].total_cost > popt[j + 1].total_cost:
                popt[j], popt[j + 1] = popt[j + 1], popt[j]
    return popt


def crossover(s1: Solution, s2: Solution) -> Solution:
    shortest = problem.shortest_path
    routes1 = s1.routes
    routes2 = s2.routes
    r1_random, r2_random = random.randint(0, len(routes1) - 1), random.randint(0, len(routes2) - 1)
    r1, r2 = routes1[r1_random], routes2[r2_random]
    r1_path, r2_path = r1.path, r2.path
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
    # print(solution2format(s1))
    # print(solution2format(s2))
    # print(solution2format(sx))
    # print("################")
    return sx


# 可能有关于元祖的bug
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
    best_load = -1
    path = route.path
    path_cost = route.cost
    path_load = route.demand
    head, tail = edges[0][0], edges[-1][1]
    edge_cost = find_parttern_cost(edges)
    edge_load = find_parttern_load(edges)
    if route.cost == 0:
        return 0, shortest[depot, head] + edge_cost + shortest[tail, depot], edge_load
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
        if temp_path_cost < best_cost and temp_path_load > best_load:
            best_position = i
            best_cost = temp_path_cost
            best_load = temp_path_load
    return best_position, best_cost, best_load


def sbx(lack: set, offspring: Route, routes: list, total_cost: int, total_load: int,
        r1_index: int) -> Solution:
    capacity = problem.capacity
    graph = problem.graph
    for edge in lack:
        best_index = None
        best_route = None
        best_cost = float('inf')
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
                temp_index, temp_cost, temp_load = find_best_position([edge], route)
                if temp_cost < best_cost and temp_load > best_load:
                    best_index, best_cost, best_load = temp_index, temp_cost, temp_load
                    best_route = route
                    new_route_index = i
        total_cost = total_cost - best_route.cost + best_cost
        total_load = total_load - best_route.demand + best_load
        if best_route != offspring:
            best_path = best_route.path.copy()
            best_path.insert(best_index, edge)
            new_route = Route(best_path, best_cost, best_load)
            routes[new_route_index] = new_route
        else:
            best_route.path.insert(best_index, edge)
            best_route.cost, best_route.demand = best_cost, best_load
    routes[r1_index] = offspring
    load = 0
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
                posit, t_cost, t_load = find_best_position([edge], candidate_route)
                reverse = False
                if t_cost < best_cost and t_load <= capacity:
                    best_cost, best_load = t_cost, t_load
                    best_origin_cost, best_origin_load = route.cost, route.demand
                    best_i, best_j = i, posit
                    best_i_route, best_j_route = i_route, j_route
                    best_reverse = reverse
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
    pass


def swap(sx: Solution) -> Solution:
    pass


def local_search(sx: Solution) -> Solution:
    best_single = single_insertion(sx)
    best_double = double_insertion(sx)
    best_swap = swap(sx)
    return best_single


def means(psize: int, opsize: int, ubtrial: int, pls: float) -> Solution:
    pop = []
    pop_set = set()
    start = time.time()
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
        # if pop[0].total_cost == best_c:
        #     return pop[0]
        if time.time() - start > termination - 1:
            return pop[0]


def resolve_problem() -> str:
    start = time.time()
    s = means(30, 180, 50, 0.2)
    print(solution2format(s))
    result = solution2format(s)
    return result


def find_route_cost(route: Route) -> int:
    pre, next = (0, 0), (0, 0)
    graph = problem.graph
    shortest = problem.shortest_path
    t_cost = 0
    for edge in route.path:
        t_cost += shortest[pre[1], edge[0]] + graph[edge].cost
        pre = edge
    t_cost += shortest[pre[1], next[0]]
    return t_cost


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
    instance_file = sys.argv[1]
    termination = int(sys.argv[3])
    random_seed = int(sys.argv[5])
    random.seed(random_seed)
    problem = Problem(instance_file)
    resolve_problem()
    exit(0)
