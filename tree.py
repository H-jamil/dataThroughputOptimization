# @Author: jamil
# @Date:   2021-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-07-04T14:44:57-05:00

import math
import random
import numpy as np
import re
import sys
import pandas as pd
from collections import Counter
from collections import deque

class Log:
    def __init__(self, serialNo, values):
        # each range is left inclusive and right exclusive, i.e., [left, right)
        self.serialNo = serialNo
        self.values = values
        # print(self.values)
        self.ranges=[]
        for value in values:
            self.ranges.append(value)
            self.ranges.append(value+1)
        # print(self.values, type(self.values))
        self.names = ['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput']

    def is_intersect(self, dimension, left, right):
        return self.values[dimension]>=left and self.values[dimension]<=right

    def is_intersect_multi_dimension(self, ranges):
        for i in range(5):
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    def matches(self, packet):
        assert len(packet) == 5, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # src ip
            packet[0] + 1,
            packet[1] + 0,  # dst ip
            packet[1] + 1,
            packet[2] + 0,  # src port
            packet[2] + 1,
            packet[3] + 0,  # dst port
            packet[3] + 1,
            packet[4] + 0,  # protocol
            packet[4] + 1
        ])

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            # result += " %s:%.2f:[%d, %d) " % (self.names[i], self.values[i], self.ranges[i * 2],
            #                             self.ranges[i * 2 + 1])
            result += " %s:%.2f" % (self.names[i], self.values[i])
        return result

def extractRequiredColumn(df,requiredFields):
    return df[df.columns[df.columns.isin(requiredFields)]]

def load_dataset_from_file(dataset_file_location):
    result_df=pd.read_csv(dataset_file_location[0])
    if len(dataset_file_location)>1:
        for i in range(1,len(dataset_file_location)):
            temp_df=pd.read_csv(dataset_file_location[i])
            result_df=pd.concat([result_df, temp_df], axis=0, join='inner')
    return result_df

class Edge:
    def __init__(self,parent_node,child_node,attr_selected,attr_value_range):
        self.parent_node=parent_node
        self.child_node=child_node
        self.attr_selected=attr_selected
        self.attr_value_range=attr_value_range

    def __str__(self):
        result="parent "+str(self.parent_node.id)+ " child "+str(self.child_node.id) + " attribute " +str(self.attr_selected)+ \
                " attribute range "+str(self.attr_value_range[0])+","+str(self.attr_value_range[1])
        return result

class Node:
    def __init__(self,id,parent,ranges,logs,depth,attribute,cut_dimension):
        self.names = ['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput']
        self.id = id
        self.parent=parent
        self.ranges = ranges
        self.logs = logs
        self.depth = depth
        self.children = []
        self.num_logs = len(self.logs)
        self.leaf_node=False
        if attribute <0:
            self.cut_attribute=None
        else:
            self.cut_attribute=self.names[attribute]
        self.edges=[]
        self.cut_dimension=cut_dimension

    def get_df(self):
        #self.names = ['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize','Parallelism', 'Concurrency','Pipelining', 'Throughput']
        self.names = ['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt']
        self.log_values=[]
        for l in self.logs:
            self.log_values.append(l.values[0:5])
        self.logs_df=pd.DataFrame(self.log_values)
        self.logs_df.columns=self.names
        return self.logs_df

    def get_max_throughput_log(self):
        throughput_log_values=[]
        for l in self.logs:
            throughput_log_values.append(l.values[10])
        max_throughput_log_values=max(throughput_log_values)
        for l in self.logs:
            if l.values[10]==max_throughput_log_values:
                return l

    def match(self, packet):

        if self.children:
            for n in self.children:
                if n.contains(packet):
                    return n.match(packet)
            return None
        else:
            for r in self.logs:
                if r.matches(packet):
                    return r

    def is_intersect_multi_dimension(self, ranges):
        for i in range(5):
            if ranges[i*2] >= self.ranges[i*2+1] or \
                    ranges[i*2+1] <= self.ranges[i*2]:
                return False
        return True

    def contains(self, packet):
        assert len(packet) == 5, packet
        return self.is_intersect_multi_dimension([
            packet[0] + 0,  # src ip
            packet[0] + 1,
            packet[1] + 0,  # dst ip
            packet[1] + 1,
            packet[2] + 0,  # src port
            packet[2] + 1,
            packet[3] + 0,  # dst port
            packet[3] + 1,
            packet[4] + 0,  # protocol
            packet[4] + 1
        ])


    def __str__(self):
        result = "ID:%d\tparent:%d\tCut_attribute:%s\tDepth:%d\tRange:\t%s\n#ofLogs:%d\nChildren: " % (
            self.id,self.parent,str(self.cut_attribute),self.depth, str(self.ranges),len(self.logs))
        for child in self.children:
            result += str(child.id) + " "
        # result += "\nlogs:\n"
        # for log in self.logs:
        #     result += str(log) + "\n"
        return result

def get_the_non_overlapping_ranges(range_left,range_right,range_per_cut,cut_num):
    ranges=[]
    for i in range(cut_num):
        if i==0:
            ranges.append(range_left+ i * range_per_cut)
            ranges.append(min(range_right, (range_left + (i + 1) * range_per_cut)))
        else:
            ranges.append(range_left+ i * range_per_cut+0.00001)
            ranges.append(min(range_right, (range_left + (i + 1) * range_per_cut)))
    return ranges

def same_values_on_dimensionX(node,cut_dimension):
    values_to_be_considered=[]
    unique_value_list=[]
    for log in node.logs:
        values_to_be_considered.append(log.values[cut_dimension])
    for value in values_to_be_considered:
        if value not in unique_value_list:
            unique_value_list.append(value)
    if len(unique_value_list)>1:
        # print (False)
        return False
    else:
        # print(True)
        return True

def standardDeviation(normalizedList):
    mean = sum(normalizedList) / len(normalizedList)
    variance = sum([((x - mean) ** 2) for x in normalizedList]) / len(normalizedList)
    sd = variance ** 0.5
    return sd

def ranked_SD_all_dimension(df):
    sdindex=[]
    df_norm = df/df.max()
    dimension_pointers=[0,1,2,3,4]
    # print(df)
    for i in list(df.columns):
        nw_list = df_norm[i].tolist()
        # print (nw_list,standardDeviation(nw_list))
        sdindex.append(standardDeviation(nw_list))
    sd_dictionary={dimension_pointers[i]: sdindex[i] for i in range(len(sdindex))}
    return {k: v for k, v in sorted(sd_dictionary.items(), key=lambda item: item[1])}

def diversityIndex(normalizedList):
    occuranceValue=0
    normMaxvalue=max(normalizedList)
    normMinvalue=min(normalizedList)
    #res is a list without the duplicates
    res = [i for n, i in enumerate(normalizedList) if i not in normalizedList[:n]] # removing the duplicates
    d = Counter(normalizedList)
    for x in res:
        occuranceValue+=1/d[x]
    diversityIn=(occuranceValue)*((normMaxvalue-normMinvalue))
    return diversityIn


def ranked_diversityIndex_all_dimension(df):
    diindex=[]
    df_norm = df/df.max()
    dimension_pointers=[0,1,2,3,4]
    for i in list(df.columns):
        nw_list = df_norm[i].tolist()
        diindex.append(diversityIndex(nw_list))
    di_dictionary={dimension_pointers[i]: diindex[i] for i in range(len(diindex))}
    return {k: v for k, v in sorted(di_dictionary.items(), key=lambda item: item[1])}

def log_value_in_node_range(search_value,attr_value_range):
    if attr_value_range[0]<=search_value and attr_value_range[1]>=search_value:
        # print("Match found")
        return True
    else:
        return False

class Tree:
    def __init__(
            self,
            logs,
            leaf_threshold,parameter_ranges,node_cutting_mechanism):

            self.leaf_threshold=leaf_threshold
            self.logs=logs
            self.node_cutting_mechanism=node_cutting_mechanism
            self.parameter_ranges=parameter_ranges
            self.root=Node(0,0,self.parameter_ranges,self.logs,0,-1,None)
            self.current_node = self.root
            self.nodes_to_cut = [self.root]
            self.depth = 0
            self.node_count = 1

    def preorderTraversal(self):
        Stack = deque([])
        # 'Preorder'-> contains all the
        # visited nodes.
        Preorder =[]
        Preorder.append(self.root)
        Stack.append(self.root)
        while len(Stack)>0:
            # 'Flag' checks whether all the child
            # nodes have been visited.
            flag = 0
            # CASE 1- If Top of the stack is a leaf
            # node then remove it from the stack:
            if len((Stack[len(Stack)-1]).children)== 0:
                Par = Stack.pop()
                # CASE 2- If Top of the stack is
                # Parent with children:
            else:
                Par = Stack[len(Stack)-1]
            # a)As soon as an unvisited child is
            # found(left to right sequence),
            # Push it to Stack and Store it in
            # Auxillary List(Marked Visited)
            # Start Again from Case-1, to explore
            # this newly visited child
            for i in range(0, len(Par.children)):
                if Par.children[i] not in Preorder:
                    flag = 1
                    Stack.append(Par.children[i])
                    Preorder.append(Par.children[i])
                    break;
                    # b)If all Child nodes from left to right
                    # of a Parent have been visited
                    # then remove the parent from the stack.
            if flag == 0:
                Stack.pop()

        return Preorder

    def create_node(self, id,parent, ranges, logs, depth,attribute,cut_dimension):
        node = Node(id,parent,ranges,logs,depth,attribute,cut_dimension)
        return node

    def get_depth(self):
        return self.depth

    def get_current_node(self):
        return self.current_node

    def is_leaf(self,node,cut_dimension):
        return (len(node.logs) <= self.leaf_threshold) or (same_values_on_dimensionX(node,cut_dimension))

    def is_leaf_only_node(self,node):
        if node.cut_dimension==None:
            return False
        else:
            cut_dimension=node.cut_dimension
            return (len(node.logs) <= self.leaf_threshold) or (same_values_on_dimensionX(node,cut_dimension))

    def is_finish(self):
        return len(self.nodes_to_cut) == 0

    def get_next_node(self):
        self.nodes_to_cut.pop()
        if len(self.nodes_to_cut) > 0:
            self.current_node = self.nodes_to_cut[-1]
        else:
            self.current_node = None
        return self.current_node

    def match(self, packet):
        return self.root.match(packet)

    def update_tree(self, node, children):
        node.children.extend(children)
        children.reverse()
        self.nodes_to_cut.pop()
        self.nodes_to_cut.extend(children)
        self.current_node = self.nodes_to_cut[-1]
        # print("current_node %d"%self.current_node.id)

    def cut_node(self, node, cut_dimension, cut_num):
        self.depth = max(self.depth, node.depth + 1)
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        #range_per_cut = math.ceil((range_right - range_left) / cut_num)
        range_per_cut = (range_right - range_left) / cut_num
        child_cut_dimension_ranges=get_the_non_overlapping_ranges(range_left,range_right,range_per_cut,cut_num)
        children = []
        children_edges=[]

        assert cut_num > 0, (cut_dimension, cut_num)
        for i in range(cut_num):
            child_ranges = list(node.ranges)
            child_ranges[cut_dimension * 2] = child_cut_dimension_ranges[2*i]
            child_ranges[cut_dimension * 2 + 1] = child_cut_dimension_ranges[2*i+1]
            child_logs = []
            for log in node.logs:
                if log.is_intersect(cut_dimension,
                                     child_ranges[cut_dimension * 2],
                                     child_ranges[cut_dimension * 2 + 1]):
                    child_logs.append(log)

            if (len(child_logs)>0):    # this is to ensure no node is created with zero logs, reduce the number of nodes
                child = self.create_node(self.node_count,node.id,child_ranges,
                                         child_logs, node.depth + 1,cut_dimension,cut_dimension)
                edge=Edge(node,child,cut_dimension,[child_ranges[cut_dimension * 2],child_ranges[cut_dimension * 2+1]])
                children.append(child)
                children_edges.append(edge)
                self.node_count += 1

        node.edges=children_edges
        self.update_tree(node, children)
        return children


    def compute_result(self):

        # memory space
        # non-leaf: 2 + 16 + 4 * child num
        # leaf: 2 + 16 * log num
        # details:
        #     header: 2 bytes
        #     region boundary for non-leaf: 16 bytes
        #     each child pointer: 4 bytes
        #     each log: 16 bytes
        result = {"bytes_per_log": 0, "memory_access": 0, \
            "num_leaf_node": 0, "num_nonleaf_node": 0, "num_node": 0}
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                next_layer_nodes.extend(node.children)

                # compute bytes per rule
                if self.is_leaf_only_node(node):
                    result["bytes_per_log"] += 2 + 32 * len(node.logs)
                    result["num_leaf_node"] += 1
                else:
                    result["bytes_per_log"] += 2 + 32 + 4 * len(node.children)
                    result["num_nonleaf_node"] += 1

            nodes = next_layer_nodes

        result["memory_access"] = self._compute_memory_access(self.root)
        result["bytes_per_log"] = result["bytes_per_log"] / len(self.logs)
        result[
            "num_node"] = result["num_leaf_node"] + result["num_nonleaf_node"]
        return result

    def _compute_memory_access(self, node):
        if self.is_leaf_only_node(node) or not node.children:
            return 1
        # if node.is_partition():
        #     return sum(self._compute_memory_access(n) for n in node.children)
        else:
            return 1 + max(
                self._compute_memory_access(n) for n in node.children)

    def get_stats(self):
        widths = []
        dim_stats = []
        nodes = [self.root]
        while len(nodes) != 0 and len(widths) < 30:
            dim = [0] * 5
            next_layer_nodes = []
            for node in nodes:
                next_layer_nodes.extend(node.children)
                # if node.action and node.action[0] == "cut":
                if node.cut_dimension==None:
                    continue
                dim[node.cut_dimension] += 1
            widths.append(len(nodes))
            dim_stats.append(dim)
            nodes = next_layer_nodes
        return {
            "widths": widths,
            "dim_stats": dim_stats,
        }

    def stats_str(self):
        stats = self.get_stats()
        out = "widths" + "," + ",".join(map(str, stats["widths"]))
        out += "\n"
        for i in range(len(stats["dim_stats"][0])):
            out += "dim{}".format(i) + "," + ",".join(
                str(d[i]) for d in stats["dim_stats"])
            out += "\n"
        return out

    def print_stats(self):
        print(self.stats_str())

    def print_layers(self, layer_num=5):
        nodes = [self.root]
        for i in range(layer_num):
            if len(nodes) == 0:
                return

            print("Layer", i)
            next_layer_nodes = []
            for node in nodes:
                print(node)
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes

    def search(self,node,log,result_node):
        return_node=result_node
        if not self.is_leaf_only_node(node):
            for edge in node.edges:
                search_value=log.values[edge.attr_selected]
                if log_value_in_node_range(search_value,edge.attr_value_range):
                    return_node=edge.child_node
                    break
            return self.search(return_node,log,return_node)
        else:
            return return_node

    def __str__(self):
        result = ""
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                result += "ID:%d\tparent:%d\tCut_attribute:%s\tDepth:%d\tRange:%s\n#ofLogs:%d\t\nChildren: [" % (
                    node.id,node.parent,str(node.cut_attribute),node.depth, str(node.ranges),len(node.logs))
                # result += "%d; %s; %s; [" % (node.id, str(node.action),
                #                              str(node.ranges))
                for child in node.children:
                    result += str(child.id) + " "
                result += "]\n"
                # result += "\nlogs:\n"
                # for log in node.logs:
                #     result += str(log) + "\n"
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes
        return result
