##
# @author Jamil Hasibul <mdhasibul.jamil@siu.edu>
 # @file Description
 # @desc Created on 2021-07-27 11:59:33 pm
 # @copyright
 #
# @Author: jamil
# @Date:   2021-06-11T18:06:03-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-07-04T15:06:00-05:00

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
import sys
sys.setrecursionlimit(10000)


parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
    type=lambda expr: [
        os.path.abspath("../DataFiles/Experimental_Log_File_Results/{}".format(r)) for r in expr.split(",")],
    default="test_log.csv",
    help="dataset file name or list of dataset files separated by comma.")

class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):
        self.logs=[]
        self.test_logs=[]
        #dataset_file_location is a "list" of considered dataset files
        self.dataset=load_dataset_from_file(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        # print(self.dataset)
        for index, row in self.requiredData.iterrows():
            self.logs.append(
            Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput']]))#'P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput'

        self.grouped_df=self.requiredData.groupby(['FileCount', 'AvgFileSize','BufSize', 'Bandwidth', 'AvgRtt'])
        # print(type(self.grouped_df))
        self.map_for_tuple_to_throughput=dict()
        for key,item in self.grouped_df:
          a_group=self.grouped_df.get_group(key)
          # print(a_group, type(a_group),'\n')
          group_max_throughput=a_group['TotalAvgTput'].max()
          self.map_for_tuple_to_throughput[key]=group_max_throughput
          number_of_rows=a_group.shape[0]
        #   print(a_group.shape)
        #   print("number_of_rows",number_of_rows)
          selected_no_test_rows=math.ceil(number_of_rows*0.3)  #30% is test data
        #   print("selected_no_test_rows", selected_no_test_rows)
          a_group_test=a_group.sample(n=selected_no_test_rows)
          # print(a_group_test, '\n')
          for index, row in a_group_test.iterrows():
              self.test_logs.append(Log(index,[row['FileCount'], row['AvgFileSize'],row['BufSize'],row['Bandwidth'],row['AvgRtt'],row['CC_Level'],row['P_Level'],row['PP_Level'],row['numActiveCores'],row['frequency'],row['TotalAvgTput']]))
        #self.test_logs=self.logs
    def return_map_for_tuple_to_throughput(self):
      return self.map_for_tuple_to_throughput

    def return_test_logs(self):
      return self.test_logs


def tree_building_with_metric_result_generation(tree,metric,test_logs,optimal_throughput_dictionary):
    if metric=="DI":
        #cut_dimension=4
        ranked_cut_dimension=ranked_diversityIndex_all_dimension(tree.current_node.get_df())
        print("ranked_cut_dimension=",ranked_cut_dimension)
        cut_dimension=list(ranked_cut_dimension)[-1]
        cut_num=8
        nodes_to_operate=[]
        # for i in tree.nodes_to_cut:
        #     nodes_to_operate.append(i.id)
        # # print("nodes to cut:",nodes_to_operate)
        while len(tree.nodes_to_cut)!=0:
            if not tree.is_leaf(tree.current_node,cut_dimension):
                print("cutting node %d now" %tree.current_node.id)
                ranked_cut_dimension=ranked_diversityIndex_all_dimension(tree.current_node.get_df())
                # print(ranked_cut_dimension)
                print("ranked_cut_dimension=",ranked_cut_dimension)
                cut_dimension=list(ranked_cut_dimension)[-1]
                print ("so cutting on %d"%cut_dimension)
                print(tree.current_node)
                # print(cut_dimensions,type(cut_dimensions))
                tree.cut_node(tree.current_node,cut_dimension,cut_num)
                for edge in tree.current_node.edges:
                    print(edge)
                nodes_to_operate=[]
                for i in tree.nodes_to_cut:
                    nodes_to_operate.append(i.id)
                print("nodes to cut:",nodes_to_operate)
                ranked_cut_dimension=ranked_diversityIndex_all_dimension(tree.current_node.get_df())
                cut_dimension=list(ranked_cut_dimension)[-1]
            else:
                print("escaping node %d as a leaf node"%tree.current_node.id)
                tree.get_next_node()
                nodes_to_operate=[]
                for i in tree.nodes_to_cut:
                    nodes_to_operate.append(i.id)
                print("nodes to cut:",nodes_to_operate)



        ################################################
        #saving the DI tree in pkl format
        ################################################
        with open("DI_tree.pkl", "wb") as f:
            pickle.dump(tree, f)

        with open("DI_tree.pkl", "rb") as f:
            tree = pickle.load(f)

        print("##########################")
        print("DI tree")
        print("##########################")
        print(tree)
        print("##########################")
        # print("Tree Results:",tree.compute_result())
        # print("##########################")
        print("Tree  stats:")
        tree.print_stats()
        print("##########################")
        tree.print_layers()
        print("##########################")
        preorderedNodesID=[]
        preorderNodes=tree.preorderTraversal()
        for node in preorderNodes:
            preorderedNodesID.append(node.id)
        print("preorderedNodesID",preorderedNodesID)
        # print("##########################")
        # print("##########################")
        # print("average error for all the test log following DI scheme is =",sum(MSE_DI_list)/len(MSE_DI_list))
        #print("Tree Results:",tree.compute_result())
        print("##########################")
        print("Number of test logs %d and training logs %d" %(len(test_logs),len(tree.logs)))
        print(" leaf threshold = %d  && number of cut = %d "%(leaf_threshold,cut_num))
        print("Number of Groups = %d"%(len(optimal_throughput_dictionary.keys())))
        print("##########################")
        MSE_DI_list=[]
        max_log_DI_list=[]
        optimal_throughput_DI_list=[]
        test_logs=test_logs[0:1000]
        for log in test_logs:
            #print(log)
            node=tree.search(tree.root,log,tree.root)
            print(" %d log found in node %d"%(log.serialNo,node.id))
            max_log=node.get_max_throughput_log()
            max_log_DI_list.append(max_log)
            optimal_throughput=optimal_throughput_dictionary[(log.values[0],log.values[1],log.values[2],log.values[3],log.values[4])]
            optimal_throughput_DI_list.append(optimal_throughput)
            cluster_max_throughput=max_log.values[10]
            #print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d , pp(pipelining) %d, number of cores %d and frequency %d GHz for DI tree"%(max_log_di.values[10],max_log_di.values[6],max_log_di.values[5],max_log_di.values[7],max_log_di.values[8],max_log_di.values[9]))
            print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d and pp(pipelining) %d, number of cores %d and frequency %d GHz"%(max_log.values[10],max_log.values[6],max_log.values[5],max_log.values[7],max_log.values[8],max_log.values[9]))
            print("%d throughput is optimal"%optimal_throughput_dictionary[(log.values[0],log.values[1],log.values[2],log.values[3],log.values[4])])
            if optimal_throughput > cluster_max_throughput:
                MSE_DI_list.append((optimal_throughput-cluster_max_throughput))
            else:
                MSE_DI_list.append(0)
        print(MSE_DI_list)
        print("##########################")
        print("average error for all the test log following DI scheme is =",sum(MSE_DI_list)/len(MSE_DI_list))
        print("Tree Results:",tree.compute_result())
        print("##########################")
        print("Number of test logs %d and training logs %d" %(len(test_logs),len(tree.logs)))
        print(" leaf threshold = %d  && number of cut = %d "%(leaf_threshold,cut_num))
        print("Number of Groups = %d"%(len(optimal_throughput_dictionary.keys())))
        print("##########################")
        return max_log_DI_list,optimal_throughput_DI_list

    elif metric=="SD":
        #cut_dimension=0
        ranked_cut_dimension=ranked_SD_all_dimension(tree.current_node.get_df())
        print("ranked_cut_dimension=",ranked_cut_dimension)
        cut_dimension=list(ranked_cut_dimension)[-1]
        # cut_dimension=list(ranked_cut_dimension)[0]
        cut_num=8
        nodes_to_operate=[]
        # for i in tree.nodes_tos_cut:
        #     nodes_to_operate.append(i.id)
        # # print("nodes to cut:",nodes_to_operate)
        while len(tree.nodes_to_cut)!=0:
            if not tree.is_leaf(tree.current_node,cut_dimension):
                print("cutting node %d now" %tree.current_node.id)
                ranked_cut_dimension=ranked_SD_all_dimension(tree.current_node.get_df())
                # print(ranked_cut_dimension)
                print("ranked_cut_dimension=",ranked_cut_dimension)
                cut_dimension=list(ranked_cut_dimension)[-1]
                # cut_dimension=list(ranked_cut_dimension)[0]
                print ("so cutting on %d"%cut_dimension)
                #cut_dimension=
                # print(cut_dimensions,type(cut_dimensions))
                tree.cut_node(tree.current_node,cut_dimension,cut_num)
                for edge in tree.current_node.edges:
                    print(edge)
                nodes_to_operate=[]
                for i in tree.nodes_to_cut:
                    nodes_to_operate.append(i.id)
                print("nodes to cut:",nodes_to_operate)
                ranked_cut_dimension=ranked_SD_all_dimension(tree.current_node.get_df())
                cut_dimension=list(ranked_cut_dimension)[-1]
            else:
                print("escaping node %d as a leaf node"%tree.current_node.id)
                tree.get_next_node()
                nodes_to_operate=[]
                for i in tree.nodes_to_cut:
                    nodes_to_operate.append(i.id)
                print("nodes to cut:",nodes_to_operate)


        ################################################
        #saving the SD tree in pkl format
        ################################################
        with open("SD_tree.pkl", "wb") as f:
            pickle.dump(tree, f)

        with open("SD_tree.pkl", "rb") as f:
            tree = pickle.load(f)

        print("##########################")
        print("SD tree")
        print("##########################")
        print(tree)
        print("##########################")
        # print("Tree Results:",tree.compute_result())
        # print("##########################")
        print("Tree  stats:")
        tree.print_stats()
        print("##########################")
        tree.print_layers()
        print("##########################")
        preorderedNodesID=[]
        preorderNodes=tree.preorderTraversal()
        for node in preorderNodes:
            preorderedNodesID.append(node.id)
        print("preorderedNodesID",preorderedNodesID)
        # print("##########################")
        # print("##########################")
        # print("average error for all the test log following DI scheme is =",sum(MSE_DI_list)/len(MSE_DI_list))
        # print("Tree Results:",tree.compute_result())
        # print("##########################")
        print("Number of test logs %d and training logs %d" %(len(test_logs),len(tree.logs)))
        print(" leaf threshold = %d  && number of cut = %d "%(leaf_threshold,cut_num))
        print("Number of Groups = %d"%(len(optimal_throughput_dictionary.keys())))
        print("##########################")
        MSE_SD_list=[]
        max_log_SD_list=[]
        optimal_throughput_SD_list=[]
        test_logs=test_logs[0:1000]
        # for i in test_logs:
        #     print(i)
        for log in test_logs:
            node=tree.search(tree.root,log,tree.root)
            # print(log)
            print(" %d log found in node %d"%(log.serialNo,node.id))
            max_log=node.get_max_throughput_log()
            max_log_SD_list.append(max_log)
            optimal_throughput=optimal_throughput_dictionary[(log.values[0],log.values[1],log.values[2],log.values[3],log.values[4])]
            optimal_throughput_SD_list.append(optimal_throughput)
            cluster_max_throughput=max_log.values[10]
            print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d and pp(pipelining) %d, number of cores %d and frequency %d GHz"%(max_log.values[10],max_log.values[6],max_log.values[5],max_log.values[7],max_log.values[8],max_log.values[9]))
            #print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d and pp(pipelining) %d"%(max_log.values[8],max_log.values[5],max_log.values[6],max_log.values[7]))
            print("%d throughput is optimal"%optimal_throughput_dictionary[(log.values[0],log.values[1],log.values[2],log.values[3],log.values[4])])
            if optimal_throughput > cluster_max_throughput:
                MSE_SD_list.append((optimal_throughput-cluster_max_throughput))
            else:
                MSE_SD_list.append(0)
        print(MSE_SD_list)
        print("##########################")
        print("average error for all the test log following SD scheme is =",sum(MSE_SD_list)/len(MSE_SD_list))
        print("Tree Results:",tree.compute_result())
        print("##########################")
        print("Number of test logs %d and training logs %d" %(len(test_logs),len(tree.logs)))
        print(" leaf threshold = %d  && number of cut = %d "%(leaf_threshold,cut_num))
        print("Number of Groups = %d"%(len(optimal_throughput_dictionary.keys())))
        print("##########################")
        return max_log_SD_list,optimal_throughput_SD_list




if __name__ == "__main__":
    args = parser.parse_args()
    #requiredFields is a list of the fields containing attributes and the output
#     requiredFields=['FileSize', 'FileCount',
#    'Bandwidth', 'RTT', 'BufferSize', 'Parallelism', 'Concurrency',
#    'Pipelining', 'Throughput']
    requiredFields=['FileCount','AvgFileSize','BufSize','Bandwidth','AvgRtt','CC_Level','P_Level','PP_Level','numActiveCores','frequency','TotalAvgTput']
    LabelName='TotalAvgTput'
   #  dropFeatureList=['Parallelism', 'Concurrency',
   # 'Pipelining']
   # fileData is an object of ReadFile class
    cut_num=8
    fileData=ReadFile(args.dataset,requiredFields)
    optimal_throughput_dictionary=fileData.return_map_for_tuple_to_throughput()
    test_logs=fileData.return_test_logs()
    print(len(test_logs))
    ranges=[]
    leaf_threshold=100
    for i in requiredFields:
        ranges.append(fileData.dataset[i].min())
        ranges.append(fileData.dataset[i].max())

    tree=Tree(fileData.logs,leaf_threshold,ranges,"DI")
    max_log_DI_list,optimal_throughput_DI_list=tree_building_with_metric_result_generation(tree,"DI",test_logs,optimal_throughput_dictionary)

    tree2=Tree(fileData.logs,leaf_threshold,ranges,"SD")
    max_log_SD_list,optimal_throughput_SD_list=tree_building_with_metric_result_generation(tree2,"SD",test_logs,optimal_throughput_dictionary)

    MSE_DI_SD_list=[]
    # max_log_DI_list=[]
    # optimal_throughput_DI_list=[]
    test_logs=test_logs[0:1000]
    for log in test_logs:
        node_di=tree.search(tree.root,log,tree.root)
        node_sd=tree2.search(tree2.root,log,tree2.root)
        print(" %d log found in node %d in DI tree and node %d in SD tree"%(log.serialNo,node_di.id,node_sd.id))
        max_log_di=node_di.get_max_throughput_log()
        max_log_sd=node_sd.get_max_throughput_log()
        # max_log_DI_list.append(max_log)
        optimal_throughput=optimal_throughput_dictionary[(log.values[0],log.values[1],log.values[2],log.values[3],log.values[4])]
        # optimal_throughput_DI_list.append(optimal_throughput)
        cluster_max_throughput=(max_log_di.values[10]+max_log_sd.values[10])/2
        print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d , pp(pipelining) %d, number of cores %d and frequency %d GHz for DI tree"%(max_log_di.values[10],max_log_di.values[6],max_log_di.values[5],max_log_di.values[7],max_log_di.values[8],max_log_di.values[9]))
        print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d , pp(pipelining) %d, number of cores %d and frequency %d GHz for SD tree"%(max_log_sd.values[10],max_log_sd.values[6],max_log_sd.values[5],max_log_sd.values[7],max_log_sd.values[8],max_log_sd.values[9]))
        print("%d throughput is achievable with p(Parallelism) %d, cc(concurrency) %d , pp(pipelining) %d, number of cores %d and frequency %d GHz for DI+SD tree"%(round((max_log_sd.values[10]+max_log_sd.values[10])/2),round((max_log_sd.values[6]+max_log_sd.values[6])/2),round((max_log_sd.values[5]+max_log_sd.values[5])/2),round((max_log_sd.values[7]+max_log_sd.values[7])/2),round((max_log_sd.values[8]+max_log_sd.values[8])/2),round((max_log_sd.values[9]+max_log_sd.values[9])/2)))
        print("%d throughput is optimal"%optimal_throughput_dictionary[(log.values[0],log.values[1],log.values[2],log.values[3],log.values[4])])
        if optimal_throughput > cluster_max_throughput:
            MSE_DI_SD_list.append((optimal_throughput-cluster_max_throughput))
        else:
            MSE_DI_SD_list.append(0)
    print(MSE_DI_SD_list)
    print("##########################")
    print("average error for all the test log following SD+DI scheme is =",sum(MSE_DI_SD_list)/len(MSE_DI_SD_list))
    print("Tree DI Results:",tree.compute_result())
    print("Tree SD Results:",tree2.compute_result())
    print("##########################")
    print("Number of test logs %d and training logs %d" %(len(test_logs),len(tree.logs)))
    print(" leaf threshold = %d  && number of cut = %d "%(leaf_threshold,cut_num))
    print("Number of Groups = %d"%(len(optimal_throughput_dictionary.keys())))
    print("##########################")

