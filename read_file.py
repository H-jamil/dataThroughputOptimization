# @Author: jamil
# @Date:   2021-04-15T13:46:34-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-04-15T13:46:57-05:00


import argparse
import os
import pandas as pd
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
    # return self.dataset[[requiredFields]]
    return df[df.columns[df.columns.isin(requiredFields)]]

def NormalizeData(df):
    #min-max normalization
    # return (df - df.min()) / (df.max() - df.min())
    # max normalization
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #       # more options can be specified also
    #     print(df)

    # the replace operation is done with whole dataFrame
    df_no_NA=df.replace(to_replace="na",value= 1).astype(float) # line replace any 'na' in the dataset with 0

    # df_no_NA=pd.to_numeric(df_no_NA)
    #df=df.map({'na':1})
    # mymap = {'na':1, 'N/A':1}
    # df_no_NA=df.applymap(lambda s: mymap.get(s) if s in mymap else s)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #       # more options can be specified also
    #     print(df_no_NA)
    # print("MAX:",df_no_NA.max())
    # print("ABS:",df_no_NA.abs())
    return df_no_NA/df_no_NA.max()


class ReadFile:
    def __init__(self,
                dataset_file_location,requiredFields):

        #dataset_file_location is a "list" of considered dataset files
        self.dataset=load_dataset_from_file(dataset_file_location)
        self.requiredData=extractRequiredColumn(self.dataset,requiredFields)
        self.requiredNormalizedData=NormalizeData(self.requiredData)


if __name__ == "__main__":
    args = parser.parse_args()
    #requiredFields is a list of the fields containing attributes and the output
    requiredFields=['FileSize', 'FileCount',
   'Bandwidth', 'RTT', 'BufferSize', 'Parallelism', 'Concurrency',
   'Pipelining', 'Throughput']

   # fileData is an object of ReadFile class

    fileData=ReadFile(args.dataset,requiredFields)

    # print(fileData.dataset.columns)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
          # more options can be specified also
        print(fileData.requiredData)
    # print(fileData.requiredData.columns)
    # print(type(fileData.requiredData.iloc[0]['Concurrency']))
    # print(type(fileData.requiredData.iloc[1]['Concurrency']))
    # print(type(fileData.requiredData.iloc[1]['BufferSize']))
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
          # more options can be specified also
        print(fileData.requiredNormalizedData)
