# @Author: jamil
# @Date:   2021-06-09T13:42:58-05:00
# @Last modified by:   jamil
# @Last modified time: 2021-06-09T14:55:58-05:00

import re
import pandas as pd
import matplotlib.pyplot as plt
import math
from matplotlib.ticker import StrMethodFormatter
from collections import Counter

def diversityIndex(normalizedList):
    occuranceValue=0
    # maxvalue=max(inputList)
    # normalizedList=[x / maxvalue for x in inputList]
    normMaxvalue=max(normalizedList)
    normMinvalue=min(normalizedList)
    #res is a list without the duplicates
    res = [i for n, i in enumerate(normalizedList) if i not in normalizedList[:n]] # removing the duplicates
    #print(normalizedList)
    #print(res)
    d = Counter(normalizedList)
    for x in res:
        #print('{} has occurred {} times'.format(x, d[x]))
        #occuranceValue+=x/d[x]
        occuranceValue+=1/d[x]
        # occuranceValue+=(-x/d[x])*(math.log((x/d[x]),2))

    # print("occuranceValue", occuranceValue)

    diversityIn=(occuranceValue)*((normMaxvalue-normMinvalue))
    return diversityIn

def main():
    logs1=[[100,250,10,200,10],[100,200,8,150,15],[50,150,15,250,20],[40,150,20,225,5],[150,225,15,150,8],[100,250,10,200,10],[100,200,8,150,15],[50,150,15,250,20],[40,150,20,225,5],[150,225,15,150,8]]
    attributeName=["Filesize","NumberOfFiles","RTT","BufferSize","Bandwidth"]
    df = pd.DataFrame(logs1, columns = attributeName)
    diindex=[]
    # print(df)
    # print(df.info())
    #print(df.describe())
    # df_norm = df-df.mean()/(df.max()-df.min())
    # print(df_norm.describe())
    df_norm = df/df.max()
    print(df_norm.describe())
    #print(df_norm.var())
    for i in attributeName:

        nw_list = df_norm[i].tolist()
        diindex.append(diversityIndex(nw_list))
        print("diversityIn for %s is %d " %(i,diindex[-1]))

if __name__== "__main__":
    main()
