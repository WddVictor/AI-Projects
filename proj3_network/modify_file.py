import multiprocessing as mp
import argparse
import sys
import re
from queue import PriorityQueue
import numpy as np
from datetime import datetime

instance_file = './samples_base1/Gnutella30_1.txt'
seed_size = 5
diffusion_model = 'IC'
termination = 10

random_seed = datetime.now().microsecond
np.random.seed(random_seed)

worker = []

network = None


def resolve_file(path: str):
    file = open(path)
    current = file.readline()
    str_array = re.split(r'[\s]', current)
    if str_array:
        vertex_num = int(str_array[0])
        edge_num = int(str_array[1])
    else:
        print('No information')
        return
    neighbors = np.zeros(vertex_num, dtype=int)
    for i in range(edge_num):
        current = file.readline()
        str_array = re.split(r'[\s]', current)
        a, b = str_array[0:2]
        a, b = int(a) , int(b)
        neighbors[b] += 1
    weight = np.zeros(vertex_num, dtype=float)
    for i in range(vertex_num):
        if neighbors[i]:
            weight[i] = 1 / neighbors[i]
    file.close()
    file = open(path)
    file_update = open('./samples_base1/Gnutella30_1.txt', 'w+')
    current = file.readline()
    file_update.write(current)
    for i in range(edge_num):
        current = file.readline()
        str_array = re.split(r'[\s]', current)
        file_update.write('{0} {1}\n'.format(current[:-1], weight[b]))


class Network:
    def __init__(self, path: str):
        self.vertex_num = None
        self.edge_num = None
        self.graph = None
        self.nodes = []
        self.resolve_file(path)
        self.get_weight()

    def resolve_file(self, path: str):
        file = open(path)
        current = file.readline()
        str_array = re.split(r'[\s]', current)
        if str_array:
            self.vertex_num = int(str_array[0])
            self.edge_num = int(str_array[1])
            self.graph = np.zeros((self.vertex_num, self.vertex_num), dtype=float)
        else:
            print('No information')
            return
        graph = self.graph
        nodes = self.nodes
        for i in range(self.vertex_num):
            nodes.append(Node(i))
        for i in range(self.edge_num):
            current = file.readline()
            str_array = re.split(r'[\s]', current)
            a, b = str_array[0:2]
            a, b = int(a) - 1, int(b) - 1
            nodes[a].neighbors.add(b)
            nodes[b].reverse_neighbors.add(a)

    def get_weight(self):
        nodes = self.nodes
        graph = self.graph
        for i in range(self.vertex_num):
            for j in range(self.vertex_num):
                if nodes[i].neighbors:
                    graph[i, j] = 1 / len(nodes[i].neighbors)
        print('finish')


class Node:
    def __init__(self, index: int):
        self.index = index
        self.neighbors = set({})
        self.reverse_neighbors = set({})


class Worker(mp.Process):
    def __init__(self, inQ, outQ, random_seed):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        np.random.seed(random_seed)  # 如果子进程的任务是有随机性的，一定要给每个子进程不同的随机数种子，否则就在重复相同的结果了

    def run(self):
        pass


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


def estimate_IC(graph, seeds: set) -> int:
    nodes = network.nodes
    active = seeds.copy()
    fresh = seeds.copy()
    influence = len(seeds)
    while fresh:
        temp_fresh = set({})
        for current_node_index in fresh:
            current_node = nodes[current_node_index]
            neighbors = current_node.neighbors
            for neighbor in neighbors:
                if not active.__contains__(neighbor):
                    weight = graph[current_node_index, neighbor]
                    random_value = np.random.random()
                    if random_value <= weight:
                        active.add(neighbor)
                        temp_fresh.add(neighbor)
        influence += len(temp_fresh)
        fresh = temp_fresh
    return influence


def estimate_LT(graph, seeds: set) -> int:
    accumulate = np.zeros(network.vertex_num, dtype=float)
    thresholds = np.random.random(network.vertex_num)
    nodes = network.nodes
    active = seeds.copy()
    fresh = seeds.copy()
    influence = len(seeds)
    while fresh:
        temp_fresh = set({})
        for current_node_index in fresh:
            current_node = nodes[current_node_index]
            neighbors = current_node.neighbors
            for neighbor in neighbors:
                if not active.__contains__(neighbor):
                    weight = graph[current_node_index, neighbor]
                    accumulate[neighbor] += weight
                    if accumulate[neighbor] >= thresholds[neighbor]:
                        temp_fresh.add(neighbor)
                        active.add(neighbor)
        influence += len(temp_fresh)
        fresh = temp_fresh
    return influence


def ise(graph, seeds: set, model: str, times: int = 10000) -> float:
    if model == 'IC':
        total_influence = 0
        for i in range(times):
            total_influence += estimate_IC(graph, seeds)
        return total_influence / times
    else:
        total_influence = 0
        for i in range(times):
            total_influence += estimate_LT(graph, seeds)
        return total_influence / times


def celf(graph, times: int = 10000, seeds_set: set = None, seeds_range: set = None) -> (float, set):
    pq = PriorityQueue()
    if not seeds_range:
        seeds_range = range(len(graph))
    if not seeds_set:
        seeds_set = set({})
        for index in seeds_range:
            influence = ise(graph, {index}, diffusion_model, times)
            pq.put((0, -influence, -index))

    start_iteration = len(seeds_set)
    while start_iteration < seed_size:
        max_iteration, max_influence, max_index = pq.get()
        max_iteration, max_influence, max_index = -max_iteration, -max_influence, -max_index
        if max_iteration == start_iteration:
            seeds_set.add(max_index)
            start_iteration += 1
        else:
            influence = ise(graph, seeds_set.union({max_index}), diffusion_model, times)
            pq.put((-(max_iteration + 1), -(influence - max_influence), -max_index))
    return seeds_set


def get_degree(vertex: tuple):
    return vertex[0]


def diffusion_degree(pick_num: int = seed_size) -> set:
    s = set({})
    graph = network.graph
    nodes = network.nodes
    cdd = list(np.zeros(network.vertex_num, dtype=tuple))
    cd_self = np.zeros(network.vertex_num, dtype=float)
    for index in range(network.vertex_num):
        cd_self[index] = sum(graph[index])

    for index in range(network.vertex_num):
        vertex = nodes[index]
        temp_cdd_n = 0
        for neighbor in vertex.neighbors:
            temp_cdd_n += graph[index, neighbor] * cd_self[neighbor]
        cdd[index] = (temp_cdd_n + cd_self[index], index)

    for i in range(-pick_num, 0):
        cdd.sort(key=get_degree)
        candidate = cdd.pop(-1)
        candidate_index = candidate[1]
        s.add(candidate_index)
        for r_neighbor_index in nodes[candidate_index].reverse_neighbors:
            cdd[r_neighbor_index] -= graph[r_neighbor_index, candidate_index] * (1 + candidate[0])
    return s


def solve_problem():
    # seeds_set = celf(network.graph)
    # print(seeds_set)
    if diffusion_model == 'IC':
        candidate_seeds = diffusion_degree()
        print(candidate_seeds)
    else:
        pass


def change_index(path: str):
    file = open(path)
    current = file.readline()
    str_array = re.split(r'[\s]', current)
    if str_array:
        vertex_num = int(str_array[0])
        edge_num = int(str_array[1])
        graph = np.zeros((vertex_num, vertex_num), dtype=float)
    else:
        print('No information')
        return
    file_update = open('./samples_base1/Gnutella30.txt', 'w+')
    file_update.write(current)
    for i in range(edge_num):
        current = file.readline()
        str_array = re.split(r'[\s]', current)
        a, b, weight = str_array[0:3]
        a, b = int(a)+1, int(b) + 1
        file_update.write('{0} {1} {2}\n'.format(a,b,weight))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--instance_file', type=str, default=instance_file)
    parser.add_argument('-k', '--seed_size', type=str, default=seed_size)
    parser.add_argument('-m', '--diffusion_model', type=str, default=diffusion_model)
    parser.add_argument('-t', '--termination', type=int, default=termination)
    args = parser.parse_args()

    if len(sys.argv) > 1:
        instance_file = args.instance_file
        seed_size = args.seed_size
        diffusion_model = args.diffusion_model
        termination = args.termination
    change_index(instance_file)
    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    # sys.stdout.flush()
    # os._exit(0)
