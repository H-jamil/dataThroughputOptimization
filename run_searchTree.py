# @Author: jamil
# @Date:   2021-06-11T18:06:03-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-06-18T17:10:21-05:00

import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
import pydot
import matplotlib.pyplot as plt
from tree import *


parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
    type=lambda expr: [
        os.path.abspath("../DataFiles/{}".format(r)) for r in expr.split(",")],
    default="xsede_revised.csv",
    help="dataset file name or list of dataset files separated by comma.")

class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):
        self.logs=[]
        #dataset_file_location is a "list" of considered dataset files
        self.dataset=load_dataset_from_file(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        # print(self.dataset)
        for index, row in self.requiredData.iterrows():
            self.logs.append(
            Log(index,[row['FileSize'], row['FileCount'],row['Bandwidth'],row['RTT'],row['BufferSize'],row['Parallelism'],row['Concurrency'],row['Pipelining'],row['Throughput']]))

if __name__ == "__main__":
    args = parser.parse_args()
    #requiredFields is a list of the fields containing attributes and the output
    requiredFields=['FileSize', 'FileCount',
   'Bandwidth', 'RTT', 'BufferSize', 'Parallelism', 'Concurrency',
   'Pipelining', 'Throughput']
    LabelName='Throughput'
    dropFeatureList=['Parallelism', 'Concurrency',
   'Pipelining']
   # fileData is an object of ReadFile class
    fileData=ReadFile(args.dataset,requiredFields)
    ranges=[]
    # print(fileData.dataset['Concurrency'].min(),fileData.dataset['Concurrency'].max())

    for i in requiredFields:
        ranges.append(fileData.dataset[i].min())
        ranges.append(fileData.dataset[i].max())
    tree=Tree(fileData.logs,2,ranges,"DI")
    #root=Node(0,0,ranges,fileData.logs,0)
    # for i in fileData.logs:
    #     print (i)
    # print(tree)
    # print(tree.current_node.id)
    tree.cut_node(tree.current_node,0,2)

    print(tree)
    # print(root.logs_df)
