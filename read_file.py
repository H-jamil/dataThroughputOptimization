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


class ReadFile:

    def __init__(self,
                dataset_file_location):

        #dataset_file_location is a "list" of considered dataset files
        self.dataset=load_dataset_from_file(dataset_file_location)


if __name__ == "__main__":
    args = parser.parse_args()
    fileData=ReadFile(args.dataset)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(fileData.dataset)
