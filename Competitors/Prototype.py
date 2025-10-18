import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import math

# 固定随机种子
random.seed(42)
np.random.seed(42)

def construct_graph(graph_path):
    undirected_G = nx.read_edgelist(graph_path, nodetype=str, data=(('timestamp', str), ('sentiment', str)), create_using=nx.Graph()) # type: ignore
    return undirected_G


class CommunityAnalyzer:
    def __init__(self, G, community_nodes, alpha=1, beta=1, seta=1, if_part=True, if_limit_threshold=True, if_threshold_pruning=True, if_BFS=True):
        self.G = G
        self.community_nodes = community_nodes
        self.alpha = alpha
        self.beta = beta
        self.seta = seta
        self.if_part = if_part
        self.if_limit_threshold = if_limit_threshold
        self.if_threshold_pruning = if_threshold_pruning
        self.if_BFS = if_BFS

    def calculate_precision_and_error_rate(self, community_nodes, threshold_nodes):
        """计算精确度和错误率"""
        correct_count = sum(1 for node in community_nodes if node in threshold_nodes)
        precision = correct_count / len(community_nodes) if community_nodes else 0
        error_rate = (len(threshold_nodes)-correct_count) / len(threshold_nodes) if threshold_nodes else 0
        return precision, error_rate


    def objective_function(self, precision, error_rate):
        """目标函数：优选较小的阈值"""
        return self.alpha * precision - self.beta * error_rate

    def find_best_center_and_threshold(self):
        best_center_node = None
        best_threshold = None
        final_threshold_nodes = None
        best_score = -float('inf')
        center_threshold = -1
        # 获取枚举节点和划分比例
        top_nodes, proportion = self.get_top_nodes(len(self.community_nodes))
        for candidate_node in top_nodes:
            shortest_path_lengths = nx.single_source_shortest_path_length(graph, candidate_node)
            max_distance = max(shortest_path_lengths[node] for node in community_nodes if node in shortest_path_lengths)
            left, right = self.determine_threshold_range(max_distance, center_threshold, len(self.community_nodes))
            temp_best_score = 0
            if self.if_BFS:
                threshold_nodes = set([candidate_node])
                # 遍历原距离字典，按照距离分层存储节点
                distance_layered_dict = {}
                flag = True
                for target_node, distance in shortest_path_lengths.items():
                    if distance not in distance_layered_dict:
                        distance_layered_dict[distance] = []
                    distance_layered_dict[distance].append(target_node)

            for threshold in range(left, right):
                # 记忆化BFS，动态规划
                if self.if_BFS:
                    if flag:
                        new_nodes = []
                        for dist in range(1, threshold + 1):
                            if dist in distance_layered_dict:
                                new_nodes.extend(distance_layered_dict[dist])
                        flag = False
                    else:
                        new_nodes = distance_layered_dict[threshold]
                    threshold_nodes.update(new_nodes)
                else:
                    threshold_nodes = [node for node, distance in shortest_path_lengths.items() if distance <= threshold]

                precision, error_rate = self.calculate_precision_and_error_rate(self.community_nodes, threshold_nodes)
                current_score = self.objective_function(precision, error_rate)

                if current_score > best_score or (current_score == best_score and threshold < best_threshold and best_threshold is None):
                    best_precision = precision
                    final_error_rate = error_rate
                    final_threshold_nodes = threshold_nodes
                    best_score = current_score
                    best_center_node = candidate_node
                    best_threshold = threshold
                    # 使用阈值限制
                    if if_limit_threshold == True:
                        center_threshold = threshold
                # 阈值剪枝
                if if_threshold_pruning == True:
                    if current_score > temp_best_score:
                        temp_best_score = current_score
                    else:
                            break
                if precision == 1.0:
                    break

        return best_score, best_precision, final_error_rate, best_center_node, best_threshold, proportion, final_threshold_nodes

    def get_top_nodes(self, num_nodes):
        """根据社区大小确定比例并筛选符合要求的点"""
        if self.if_part and num_nodes > 100:
            # 这里可以添加代码来选择社区中的重要节点
            return self.find_top_nodes()
        else:
            return self.community_nodes, 1

    # 根据社区大小确定比例
    def determine_proportion(self, community_size, max_community_size, min_proportion=0.05):
        # 使用Sigmoid函数调整比例，使得小社区有较高的比例，大社区有较低的比例
        # 这里我们调整Sigmoid函数，使其在社区大小接近max_community_size一半时开始快速下降
        k = 5 / max_community_size  # 控制曲线的陡峭程度
        x0 = max_community_size / 2  # 控制曲线的横向移动，这里设置为最大社区大小的一半
        proportion = min(1.0, min_proportion + (1 - 1 / (1 + math.exp(-k * (community_size - x0)))))

        return proportion


    # 按比例筛选符合要求的点
    def find_top_nodes(self):
        # 根据节点的度数进行排序
        degree_dict = {node: self.G.degree(node) for node in community_nodes}
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)  # 按度数从大到小排序
        # 计算取点比例
        proportion = self.determine_proportion(len(community_nodes), max_community_size)
        # 计算要取的节点数量
        num_top_nodes = max(100, int(len(sorted_nodes) * proportion))  # 确保至少有一个节点
        # 只取度数较大的点
        top_nodes = sorted_nodes[:num_top_nodes]
        # print(f'总数：{len(community_nodes)}，取节点数：{len(top_nodes)}，划分比例：{proportion * 100:.2f}%')
        return top_nodes, proportion

    def determine_threshold_range(self, max_distance, center_threshold, num_nodes):
        """确定阈值搜索范围"""
        if self.if_limit_threshold and center_threshold != -1 and num_nodes > 100:
            left = max(1, center_threshold - self.seta)
            right = min(center_threshold + self.seta, max_distance + 1)
        else:
            left = 1
            right = max_distance + 1
        return left, right

if __name__ == '__main__':
    community_path = "./Dataset/User_Study/community.txt"
    graph_path = "./Dataset/User_Study/graph.txt"
    community = construct_graph(community_path)
    graph = construct_graph(graph_path)

    tag = [False,True]
    i = 1
    if_write = True
    #是否使用社区中全部点
    if_part = tag[i]
    # 是否限制阈值范围
    if_limit_threshold = tag[i]
    # 是否使用阈值剪枝
    if_threshold_pruning = tag[i]
    # 是否使用BFS
    if_BFS = tag[i]


    community_nodes = list(community.nodes())
    max_community_size = len(community_nodes)

    # 存储每个社区的中心节点、阈值、覆盖率等信息
    analyzer = CommunityAnalyzer(graph, community_nodes, if_part=if_part, if_limit_threshold=if_limit_threshold, if_threshold_pruning=if_threshold_pruning, if_BFS=if_BFS)
    part_start_time = time.time()
    score, coverage_rate, error_rate, center_node, threshold_distance, proportion, threshold_nodes = analyzer.find_best_center_and_threshold()
    part_end_time = time.time()
    part_total_cost_time = part_end_time - part_start_time
    print(f'Central node {center_node}，threshold radius {threshold_distance}')

    # 输出社区的结果
    # print(f"Number of nodes: {len(community_nodes)}  score: [{score:.2f}]")
    # print(f"  Central node: {center_node}, threshold radius: {threshold_distance}")
    # print(f"  Initial nodes: {community_nodes}")
    # print(f"  All nodes in the explainable region: {threshold_nodes}")
    # print(f"  Proportion：{proportion * 100:.2f}%, time consuming: {part_total_cost_time * 1000}ms")
    # print(f"  Coverage rate: {coverage_rate:.2%}, Error rate: {error_rate:.2%}\n")

    # Visualize the community
    plt.figure(figsize=(4, 4))
    pos = nx.circular_layout(community)
    # For center node, assign a larger size, mark the threshold distance within the community
    node_sizes = [1000 if node == center_node else 500 for node in community.nodes()]
    text = "r=" + str(threshold_distance)
    # Put the text in the empty space of the circular layout
    plt.text(0.2, 0.1, text, fontsize=12, ha='center', va='center', color='black')
    nx.draw(community, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=node_sizes, font_size=10)
    plt.show()