# @Author: jamil
# @Date:   2021-07-02T23:02:07-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-07-02T23:03:10-05:00



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


with open("DI_tree.pkl", "rb") as f:
    tree = pickle.load(f)
print("##########################")
print("DI tree")
print("##########################")
print(tree)
print("##########################")
print("Tree Results:",tree.compute_result())
print("##########################")
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
print("##########################")


with open("SD_tree.pkl", "rb") as f:
    tree = pickle.load(f)
print("##########################")
print("SD tree")
print("##########################")
print(tree)
print("##########################")
print("Tree Results:",tree.compute_result())
print("##########################")
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
print("##########################")
