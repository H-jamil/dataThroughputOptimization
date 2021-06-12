# @Author: jamil
# @Date:   2021-06-11T16:50:09-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-06-12T00:01:43-05:00

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
        # print(self.values, type(self.values))
        self.names = ['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize','Parallelism','Concurrency','Pipelining', 'Throughput']

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            result += " %s:%.2f " % (self.names[i], self.values[i])
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
    def __init__(self,mother_node,child_node):
        self.mother_node=mother_node
        self.child_node=child_node
        self.attr_selected=None
        self.attr_value_range=[]

    def __str__(self):
        result="mother "+str(self.mother_node.id)+ " child "+str(self.child_node.id) + "attribute " +str(attr_selected)+ \
                "attribute range "+str(self.attr_value_range[0])+","+str(self.attr_value_range[1])
        return result

class Node:
    def __init__(self, id, ranges, logs, depth):
        self.id = id
        self.ranges = ranges
        self.logs = logs
        self.depth = depth
        self.children = []
        self.num_logs = len(self.logs)
        self.leaf_node=False
        self.cut_attribute=None
        self.edges=[]
        self.names = ['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize','Parallelism', 'Concurrency','Pipelining', 'Throughput']
        self.log_values=[]
        for l in self.logs:
            self.log_values.append(l.values)
        self.logs_df=pd.DataFrame(self.log_values)
        self.logs_df.columns=self.names

    def __str__(self):
        result = "ID:%d\tCut_attribute:%s\tDepth:%d\tRange:\t%s\nChildren: " % (
            self.id,str(self.cut_attribute),self.depth, str(self.ranges))
        for child in self.children:
            result += str(child.id) + " "
        result += "\nlogs:\n"
        for log in self.logs:
            result += str(log) + "\n"
        return result
