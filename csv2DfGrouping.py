##
# @author Jamil Hasibul <mdhasibul.jamil@siu.edu>
 # @file Description
 # @desc Created on 2021-07-27 11:33:15 pm
 # @copyright
 #
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
import time
import pickle
import math



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
        self.test_logs=[]
        #dataset_file_location is a "list" of considered dataset files
        self.dataset=load_dataset_from_file(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        # print(self.requiredData)
        # print(type(self.requiredData))
        for index, row in self.requiredData.iterrows():
            self.logs.append(
            Log(index,[row['FileSize'], row['FileCount'],row['Bandwidth'],row['RTT'],row['BufferSize'],row['Parallelism'],row['Concurrency'],row['Pipelining'],row['Throughput']]))

        self.grouped_df=self.requiredData.groupby(['FileSize', 'FileCount','Bandwidth', 'RTT', 'BufferSize'])
        # print(type(self.grouped_df))
        self.map_for_tuple_to_throughput=dict()
        for key,item in self.grouped_df:
          a_group=self.grouped_df.get_group(key)
          # print(a_group, type(a_group),'\n')
          group_max_throughput=a_group['Throughput'].max()
          self.map_for_tuple_to_throughput[key]=group_max_throughput

          number_of_rows=a_group.shape[0]
          selected_no_test_rows=math.ceil(number_of_rows*0.3)  #30% is test data
          a_group_test=a_group.sample(n=selected_no_test_rows)
          # print(a_group_test, '\n')
          for index, row in a_group_test.iterrows():
            self.test_logs.append(
            Log(index,[row['FileSize'], row['FileCount'],row['Bandwidth'],row['RTT'],row['BufferSize'],row['Parallelism'],row['Concurrency'],row['Pipelining'],row['Throughput']]))

        for l in self.test_logs:
          print(l)
        print(self.map_for_tuple_to_throughput)

    def return_map_for_tuple_to_throughput(self):
      return self.map_for_tuple_to_throughput
    # def group_based_on_tuples(self,dataset_file_location,requiredFields):
    #   self.df=pd.read_csv(dataset_file_location)

    def return_test_logs(self):
      return self.test_logs



if __name__ == "__main__":
    args = parser.parse_args()
    #requiredFields is a list of the fields containing attributes and the output
    requiredFields=['FileSize', 'FileCount',
   'Bandwidth', 'RTT', 'BufferSize', 'Parallelism', 'Concurrency',
   'Pipelining', 'Throughput']
    LabelName='Throughput'
   #  dropFeatureList=['Parallelism', 'Concurrency',
   # 'Pipelining']
   # fileData is an object of ReadFile class

    fileData=ReadFile(args.dataset,requiredFields)
    optimal_throughput_dictionary=fileData.return_map_for_tuple_to_throughput()
    print(optimal_throughput_dictionary)
    for log in fileData.return_test_logs():
      print(log)
