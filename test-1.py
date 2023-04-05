# @Time    : 2023/4/5 22:04
# @Author  : ygd
# @FileName: test-1.py
# @Software: PyCharm


import matplotlib.pyplot as plt

node_nums = 1
node_nums_YC = 2
node_nums_x = 5
node_nums_QC = 3
nodes_YC = {}
nodes_QC = {}
nodes = {}
edges = []
road_length = 2
YC_interval = 2
QC_interval = 1
buffer_length = 3

# YC node
for j in range(node_nums_YC):
    for i in range(node_nums_x):
        nodes_YC[node_nums] = [i * road_length, j * YC_interval]
        plt.plot(i * road_length, j * YC_interval, 'o', color='b')
        # plt.text(i * road_length + 0.5, j * YC_interval - 0.5, f"{node_nums}", fontsize=10)
        node_nums += 1

# YC最高点
MAX_YC_Y = sorted(nodes_YC.items(), key=lambda x: x[1][1], reverse=True)[0][1][1]

# QC node
for j in range(node_nums_QC):
    for i in range(node_nums_x):
        nodes_QC[node_nums] = [i * road_length, j * QC_interval + MAX_YC_Y + buffer_length]
        plt.plot(i * road_length, j * QC_interval + MAX_YC_Y + buffer_length, 'o', color='r')
        # plt.text(i * road_length + 0.5, j * QC_interval - 0.5, f"{node_nums}", fontsize=10)
        node_nums += 1

nodes.update(nodes_YC)
nodes.update(nodes_QC)

# YC从左到右
for i in range(1, len(nodes_YC), node_nums_x):
    for j in range(i, node_nums_x + i - 1):
        start = nodes_YC[j]
        end = nodes_YC[j + 1]
        edges.append([start, end])

# YC双向
for i in range(1, node_nums_x + 1):
    for j in range(node_nums_YC - 1):
        start = nodes_YC[i + node_nums_x * j]
        end = nodes_YC[i + node_nums_x * (j + 1)]
        edges.append([start, end])
        edges.append([end, start])

# YC最大坐标序号的值
MAX_YC_Y_NODES = sorted(nodes_YC.items(), key=lambda x: x[0], reverse=True)[0][0]
print(nodes_YC)
print(nodes_QC)

# QC从右到左
for i in range(node_nums_QC):
    for j in range(i * node_nums_x + 1 + MAX_YC_Y_NODES, node_nums_x + i * node_nums_x + MAX_YC_Y_NODES):
        end = nodes_QC[j]
        start = nodes_QC[j + 1]
        edges.append([start, end])

# QC双向
for i in range(1, node_nums_x + 1):
    for j in range(node_nums_YC,
                   node_nums_YC + node_nums_QC - 1):
        start = nodes_QC[i + node_nums_x * j]
        end = nodes_QC[i + node_nums_x * (j + 1)]
        edges.append([start, end])
        edges.append([end, start])

# buffer
# 双向
for i in range(node_nums_x * (node_nums_YC - 1) + 1, node_nums_x * node_nums_YC + 1):
    start = nodes_YC[i]
    end = nodes_QC[i + node_nums_x]
    if i & 1:
        edges.append([start, end])
    else:
        edges.append([end, start])

for start, end in edges:
    plt.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=0.2, head_length=0.2,
              length_includes_head=True)

plt.show()
