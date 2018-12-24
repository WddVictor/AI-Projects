def dijkstra(graph,src):
    # 判断图是否为空，如果为空直接退出
    if graph is None:
        return None
    nodes = [i for i in range(len(graph))]  # 获取图中所有节点
    visited=[]  # 表示已经路由到最短路径的节点集合
    if src in nodes:
        visited.append(src)
        nodes.remove(src)
    else:
        return None
    distance={src:0}  # 记录源节点到各个节点的距离
    for i in nodes:
        distance[i]=graph[src][i]  # 初始化
    # print(distance)
    path={src:{src:[]}}  # 记录源节点到每个节点的路径
    k=pre=src
    while nodes:
        mid_distance=1000
        for v in visited:
            for d in nodes:
                new_distance = graph[src][v]+graph[v][d]
                if new_distance < mid_distance:
                    mid_distance=new_distance
                    graph[src][d]=new_distance  # 进行距离更新
                    k=d
                    pre=v
        distance[k]=mid_distance  # 最短路径
        path[src][k]=[i for i in path[src][pre]]
        path[src][k].append(k)
        # 更新两个节点集合
        visited.append(k)
        print(k,nodes)
        nodes.remove(k)
        print(visited,nodes)  # 输出节点的添加过程
    return distance,path
if __name__ == '__main__':
    graph_list = [[1000 , 4. ,1000 , 4.  ,3., 1000 ,1000  ,1.],
 [ 4. ,1000 , 2. ,1000 ,1000 ,1000 ,1000, 1000],
 [1000 , 2., 1000,  3. ,1000, 1000 ,1000 ,1000],
 [ 4. ,1000  ,3. ,1000 , 7. ,1000 ,1000 ,1000],
 [ 3. ,1000 ,1000  ,7. ,1000 , 2. ,1000 ,1000],
 [1000 ,1000, 1000 ,1000  ,2. ,1000 , 3. ,1000],
 [1000 ,1000, 1000 ,1000 ,1000 , 3. ,1000 , 3.],
 [ 1. ,1000, 1000 ,1000, 1000 ,1000 , 3. ,1000]]

    g_list = [[1,2],
              [2,1]]

    distance,path= dijkstra(g_list, 1)  # 查找从源点0开始带其他节点的最短路径
    print(distance,path)
