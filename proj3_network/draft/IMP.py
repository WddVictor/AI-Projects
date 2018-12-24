import argparse
import os
import random
import sys
import re
from math import *
from queue import PriorityQueue
import numpy as np
from datetime import datetime

instance_file = './samples_base1/Amazon.txt'
seed_size = 50
diffusion_model = 'IC'
termination = 20

random_seed = datetime.now().microsecond
np.random.seed(random_seed)

worker = []

network = None


class Network:
    def __init__(self, path: str):
        self.vertex_num = None
        self.edge_num = None
        self.graph = None
        self.nodes = []
        self.resolve_file(path)

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
            a, b, weight = str_array[0:3]
            a, b, weight = int(a)-1, int(b)-1, float(weight)
            graph[a, b] = weight
            nodes[a].neighbors.add(b)
            nodes[b].reverse_neighbors.add(a)


class Node:
    def __init__(self, index: int):
        self.index = index
        self.neighbors = set({})
        self.reverse_neighbors = set({})


def combination(n: int, k: int) -> float:
    return factorial(n) / (factorial(k) * factorial(n - k))


def covered_fraction(r: list, s: set) -> float:
    num = 0
    for rr_set in r:
        if s.intersection(rr_set):
            num += 1
    return num / len(r)


def celf(r: list, k: int) -> set:

    pq = PriorityQueue()
    seed_range = set({})
    sk = set({})
    seeds_label = np.zeros(network.vertex_num,dtype=int)

    for rr_set in r:
        for vertex in rr_set:
            if not seeds_label[vertex]:
                seed_range.add(vertex)
            seeds_label[vertex]+=1
    for index in seed_range:
        influence = seeds_label[index]
        pq.put((-influence, 0, index))

    iteration = 0

    while iteration < k:
        max_influence, max_iteration, max_index = pq.get()
        max_influence, max_iteration, max_index = -max_influence, -max_iteration, max_index
        if max_index not in seed_range:
            continue
        if max_iteration == iteration:
            sk.add(max_index)
            for rr_set in r:
                if max_index in rr_set:
                    for vertex in rr_set:
                        seeds_label[vertex]-=1
                        if not seeds_label[vertex]:
                            seed_range.remove(vertex)
            iteration += 1
        else:
            influence = seeds_label[max_index]
            pq.put((-influence, -iteration, max_index))

    return sk


def node_selection(r: list, k: int) -> set:
    return celf(r, k)



def get_rr_set(v: int) -> set:
    result = {v}
    graph = network.graph
    nodes = network.nodes
    q = [v]
    while q:
        # if (datetime.now() - start).seconds > termination - 5:
        #     emergency()
        next_q = []
        for current in q:
            reverse_neighbors = nodes[current].reverse_neighbors
            if diffusion_model=='IC':
                for r_neighbor in reverse_neighbors:

                    if random.random() < graph[r_neighbor, current]:
                        if r_neighbor not in result:
                            next_q.append(r_neighbor)
                result.update(set(next_q))
            else:
                r_num = random.random()
                temp = 0
                for reverse_neighbor in reverse_neighbors:
                    if temp <= r_num <= temp+graph[reverse_neighbor, current]:
                        if reverse_neighbor not in result:
                            next_q.append(reverse_neighbor)
                            result.add(reverse_neighbor)
                        break
                    temp+= graph[reverse_neighbor, current]
        q = next_q
    return result


def sampling(k: int, delta: float, l: float):
    r = []
    lb = 1
    n = network.vertex_num
    t_delta = 1.4142135623730951 * delta

    log_combination = log(combination(n, k))
    log_n = log(n)

    lamda = (2 + 2 * t_delta / 3) * (log_combination + l * log_n + log(log(n, 2))) * n / pow(t_delta, 2)
    for i in range(1, int(log(n, 2))):
        x = n / pow(2, i)
        sita = lamda / x
        while len(r) <= sita:
            v = random.randint(0, network.vertex_num - 1)
            rr = get_rr_set(v)
            r.append(rr)
        si = node_selection(r, k)
        if n * covered_fraction(r, si) >= (1 + t_delta) * x:
            lb = n * covered_fraction(r, si) / (1 + t_delta)
            break

    alpha = sqrt(l * log_n + 0.69314718055995)
    beta = sqrt((1 - 1 / e) * (log_combination + l * log_n + 0.69314718055995))
    lamda_star = 2 * n * pow((1 - 1 / e) * alpha + beta, 2) * pow(delta, -2)
    sita = lamda_star / lb
    while len(r) <= sita:
        v = random.randint(0, network.vertex_num - 1)
        rr = get_rr_set(v)
        r.append(rr)
    return r


def imm(k: int, delta: float, l: float):
    n = network.vertex_num
    l = l * (1 + 0.69314718055995 / log(n))
    r = sampling(k, delta, l)
    sk = node_selection(r, k)
    return sk

def diffusion_degree(pick_num: int = seed_size) -> set:
    s = set({})
    graph = network.graph
    nodes = network.nodes
    cdd = list(np.zeros(network.vertex_num, dtype=tuple))
    cd_self = np.zeros(network.vertex_num, dtype=float)
    for index in range(network.vertex_num):
        vertex = nodes[index]
        temp_cd_self = 0
        for neighbor in vertex.neighbors:
            temp_cd_self += graph[index, neighbor]
        cd_self[index] = temp_cd_self

    for index in range(network.vertex_num):
        vertex = nodes[index]
        temp_cdd_n = 0
        for neighbor in vertex.neighbors:
            temp_cdd_n += graph[index, neighbor] * cd_self[neighbor]
        cdd[index] = (temp_cdd_n + cd_self[index], index)

    for i in range(-pick_num, 0):
        cdd.sort(key=get_degree)
        candidate = cdd[-1]
        candidate_index = candidate[1]
        idx = -1
        while candidate_index in s:
            idx -=1
            candidate = cdd[idx]
            candidate_index = candidate[1]
        s.add(candidate_index)
        for r_neighbor_index in nodes[candidate_index].reverse_neighbors:
            cdd[r_neighbor_index] = (graph[r_neighbor_index, candidate_index] * (1 + candidate[0]),r_neighbor_index)
    return s

def get_degree(vertex):
    return vertex[0]

def emergency():
    candidate_seeds = diffusion_degree()
    for i in candidate_seeds:
        print(i+1)
    sys.stdout.flush()
    os._exit(0)

def solve_problem():
    candidate_seeds = imm(seed_size, 0.1, 1)

    for i in candidate_seeds:
        print(i+1)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--instance_file', type=str, default=instance_file)
    parser.add_argument('-k', '--seed_size', type=int, default=seed_size)
    parser.add_argument('-m', '--diffusion_model', type=str, default=diffusion_model)
    parser.add_argument('-t', '--termination', type=int, default=termination)
    args = parser.parse_args()
    start = datetime.now()
    if len(sys.argv) > 1:
        instance_file = args.instance_file
        seed_size = args.seed_size
        diffusion_model = args.diffusion_model
        termination = args.termination
    network = Network(instance_file)
    solve_problem()

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
    os._exit(0)
