# @Author: jamil
# @Date:   2021-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-06-18T17:11:06-05:00

import math
import random
import numpy as np
import re
import sys
import pandas as pd

# sys.setrecursionlimit(99999)
# SPLIT_CACHE = {}

class Log:
    def __init__(self, serialNo, values):
        # each range is left inclusive and right exclusive, i.e., [left, right)
        self.serialNo = serialNo
        self.values = values
        self.ranges=[]
        for value in values:
            self.ranges.append(value)
            self.ranges.append(value+1)
        # print(self.values, type(self.values))
        self.names = ['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize','Parallelism','Concurrency','Pipelining', 'Throughput']

    def is_intersect(self, dimension, left, right):
        # return not (left >= self.ranges[dimension*2+1] or \
        #     right <= self.ranges[dimension*2])
        return self.values[dimension]>=left and self.values[dimension]<=right

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
    def __init__(self,parent_node,child_node):
        self.parent_node=parent_node
        self.child_node=child_node
        self.attr_selected=None
        self.attr_value_range=[]

    def __str__(self):
        result="parent "+str(self.parent_node.id)+ " child "+str(self.child_node.id) + "attribute " +str(attr_selected)+ \
                "attribute range "+str(self.attr_value_range[0])+","+str(self.attr_value_range[1])
        return result

class Node:
    def __init__(self,id,parent,ranges,logs,depth,attribute):
        self.names = ['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize','Parallelism', 'Concurrency','Pipelining', 'Throughput']
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

    def get_df():
        self.names = ['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize','Parallelism', 'Concurrency','Pipelining', 'Throughput']
        self.log_values=[]
        for l in self.logs:
            self.log_values.append(l.values)
        self.logs_df=pd.DataFrame(self.log_values)
        self.logs_df.columns=self.names
        return self.logs_df

    def __str__(self):
        result = "ID:%d\tparent:%d\tCut_attribute:%s\tDepth:%d\tRange:\t%s\nChildren: " % (
            self.id,self.parent,str(self.cut_attribute),self.depth, str(self.ranges))
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
            ranges.append(range_left+ i * range_per_cut+1)
            ranges.append(min(range_right, (range_left + (i + 1) * range_per_cut)))
    return ranges

class Tree:
    def __init__(
            self,
            logs,
            leaf_threshold,parameter_ranges,node_cutting_mechanism):

            self.leaf_threshold=leaf_threshold
            self.logs=logs
            self.node_cutting_mechanism=node_cutting_mechanism
            self.parameter_ranges=parameter_ranges
            self.root=Node(0,0,self.parameter_ranges,self.logs,0,-1)
            self.current_node = self.root
            self.nodes_to_cut = [self.root]
            self.depth = 0
            self.node_count = 1

    def create_node(self, id,parent, ranges, logs, depth,attribute):
        node = Node(id,parent,ranges,logs,depth,attribute)
        return node

    def get_depth(self):
        return self.depth

    def get_current_node(self):
        return self.current_node

    def is_leaf(self, node):
        return len(node.logs) <= self.leaf_threshold

    def is_finish(self):
        return len(self.nodes_to_cut) == 0

    def update_tree(self, node, children):
        # if self.refinements["node_merging"]:
        #     children = self.refinement_node_merging(children)
        #
        # if self.refinements["equi_dense"]:
        #     children = self.refinement_equi_dense(children)
        #
        # if (self.refinements["region_compaction"]):
        #     for child in children:
        #         self.refinement_region_compaction(child)
        node.children.extend(children)
        children.reverse()
        self.nodes_to_cut.pop()
        self.nodes_to_cut.extend(children)
        self.current_node = self.nodes_to_cut[-1]

    def cut_node(self, node, cut_dimension, cut_num):
        # self.node_count += 1
        self.depth = max(self.depth, node.depth + 1)
        # node.action = ("cut", cut_dimension, cut_num)
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        range_per_cut = math.ceil((range_right - range_left) / cut_num)
        child_cut_dimension_ranges=get_the_non_overlapping_ranges(range_left,range_right,range_per_cut,cut_num)
        # print(range_left,range_right,range_per_cut)
        children = []
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
                    # print("True\n")

            # for log in child_logs:
            #     print(log)
            # print(self.node_count, type(self.node_count))
            # print(node.id,type (node.id))
            # print(child_ranges,type(child_ranges))
            # print("child_logs:",type(child_logs))
            # print(node.depth,type(node.depth))
            # if len(child_logs)==0:

            child = self.create_node(self.node_count,node.id,child_ranges,
                                     child_logs, node.depth + 1,0)
            children.append(child)
            self.node_count += 1

        self.update_tree(node, children)
        return children

    def __str__(self):
        result = ""
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                result += "ID:%d\tparent:%d\tCut_attribute:%s\tDepth:%d\tRange:%s\t#ofLogs:%d\t\nChildren: [" % (
                    node.id,node.parent,str(node.cut_attribute),node.depth, str(node.ranges),len(node.logs))
                # result += "%d; %s; %s; [" % (node.id, str(node.action),
                #                              str(node.ranges))
                for child in node.children:
                    result += str(child.id) + " "
                result += "]\n"
                result += "\nlogs:\n"
                for log in node.logs:
                    result += str(log) + "\n"
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes
        return result
