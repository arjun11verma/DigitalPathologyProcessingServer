import numpy as np 

class Edge:
    def __init__(self, node_one, node_two, weight):
        self.node_one = np.array(node_one)
        self.node_two = np.array(node_two)
        self.weight = weight