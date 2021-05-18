# @Author: jamil
# @Date:   2021-04-15T13:46:34-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-05-18T16:37:34-05:00


import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser()

parser.add_argument("--dataset",
    type=lambda expr: [
        os.path.abspath("../DataFiles/{}".format(r)) for r in expr.split(",")],
    default="xsede_revised.csv",
    help="dataset file name or list of dataset files separated by comma.")


#dataset_file_location is a "list" of considered dataset files
#function takes a list and return a dataframe created from provided locations.
#capable of creating a single dataframe for multiple data file locations

def load_dataset_from_file(dataset_file_location):
    result_df=pd.read_csv(dataset_file_location[0])
    if len(dataset_file_location)>1:
        for i in range(1,len(dataset_file_location)):
            temp_df=pd.read_csv(dataset_file_location[i])
            result_df=pd.concat([result_df, temp_df], axis=0, join='inner')
    return result_df


def extractRequiredColumn(df,requiredFields):
    return df[df.columns[df.columns.isin(requiredFields)]]

def NormalizeData(df):
    df_no_NA=df.replace(to_replace="na",value= 1).astype(float) # line replace any 'na' in the dataset with 0
    return df_no_NA/df_no_NA.max()

def set_labels(dataFrame,LabelName):
    return np.array(dataFrame[LabelName])

def set_features(dataFrame,LabelName,dropFeatureList):
    dataFrame= dataFrame.drop(LabelName, axis = 1)
    for i in dropFeatureList:
        dataFrame= dataFrame.drop(i, axis = 1)
    return dataFrame

class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):

        #dataset_file_location is a "list" of considered dataset files
        self.dataset=load_dataset_from_file(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        self.requiredNormalizedData=NormalizeData(self.requiredData)

class pd2Array:
    def __init__(self,dataFrame,LabelName,dropFeatureList):

        self.labels=set_labels(dataFrame,LabelName)
        self.features=set_features(dataFrame,LabelName,dropFeatureList)

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

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #       # more options can be specified also
    #     print(fileData.requiredNormalizedData)
    # print(fileData.requiredNormalizedData['FileSize'])


    print(fileData.requiredNormalizedData.describe())
    processedData=pd2Array(fileData.requiredNormalizedData,LabelName,dropFeatureList)
    print(processedData.labels,len(processedData.labels)) # processedData.labels is np.array
    print(processedData.features,processedData.features.shape) #processedData.features are dataFrame
