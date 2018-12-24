import multiprocessing as mp
import os
import argparse
import sys
import re
import numpy as np
from datetime import datetime

instance_file = './samples_base1/Amazon.txt'
seed_set = './seeds/seeds.txt'
diffusion_model = 'IC'
termination = 10
seeds = set({})

estimate_times = 1000

random_seed = datetime.now().microsecond
np.random.seed(random_seed)

worker_num = 8
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

class Node:
    def __init__(self, index: int):
        self.index = index
        self.neighbors = set({})


class Worker(mp.Process):
    def __init__(self, inQ, outQ, random_seed):
        super(Worker, self).__init__(target=self.start)
        self.inQ = inQ
        self.outQ = outQ
        np.random.seed(random_seed)  # 如果子进程的任务是有随机性的，一定要给每个子进程不同的随机数种子，否则就在重复相同的结果了

    def run(self):
        influence = ise(seeds,diffusion_model,int(estimate_times/worker_num))
        self.outQ.put(influence)

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


def get_seeds() -> set:
    file = open(seed_set)
    seeds = set({})
    while True:
        current = file.readline()
        str_array = re.split(r'[\s]', current)
        if current:
            seeds.add(int(str_array[0])-1)
        else:
            break
    return seeds


def estimate_IC(seeds: set) -> int:
    nodes = network.nodes
    graph = network.graph
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


def estimate_LT(seeds: set) -> int:
    accumulate = np.zeros(network.vertex_num, dtype=float)
    thresholds = np.random.random(network.vertex_num)
    nodes = network.nodes
    graph = network.graph
    active = seeds.copy()
    fresh = seeds.copy()
    for k,v in enumerate(thresholds):
        if not v:
            active.add(k)
            fresh.add(k)
    influence = len(active)
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


def ise(seeds: set, model: str,times:int=10000) -> float:
    if model == 'IC':
        total_influence = 0
        for i in range(times):
            total_influence += estimate_IC(seeds)
        return total_influence/times
    else:
        total_influence = 0
        for i in range(times):
            total_influence += estimate_LT(seeds)
        return total_influence / times

def solve_problem():
    create_worker(worker_num)
    influence = 0
    for i in range(worker_num):
        influence+=worker[i].outQ.get()
    print(influence/worker_num)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--instance_file', type=str, default=instance_file)
    parser.add_argument('-s', '--seed_set', type=str, default=seed_set)
    parser.add_argument('-m', '--diffusion_model', type=str, default=diffusion_model)
    parser.add_argument('-t', '--termination', type=int, default=termination)
    args = parser.parse_args()

    if len(sys.argv) > 1:
        instance_file = args.instance_file
        seed_set = args.seed_set
        diffusion_model = args.diffusion_model
        termination = args.termination
    network = Network(instance_file)
    seeds = get_seeds()

    solve_problem()

    '''
    程序结束后强制退出，跳过垃圾回收时间, 如果没有这个操作会额外需要几秒程序才能完全退出
    '''
    sys.stdout.flush()
    os._exit(0)
