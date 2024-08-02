# construct indexes for multiple datafiles using different configs

# Author: Buddhi Ashan M. K.
# Date: 02-06-2024

# conda environment: conda activate ssearch2
# Run with no hang up: nohup time python indexConstruct.py > indexSave/icout-pk-th_0.002.log&

# import library
from shapely.geometry.polygon import Polygon
from multiprocessing import Process, Array
from shapely.strtree import STRtree
from shapely import affinity
import matplotlib.pyplot as plt

import shapely.wkt
import math

import sys 
import nmslib 
import time 

print(sys.version)
print("NMSLIB version:", nmslib.__version__)

import os

from multiprocessing import Process, Array

# read a encoding file
def readEncodeFile(file):
    # print(file)
    f=open(file, 'r')
    lines=f.readlines()
    f.close()

    strs=[]
    for lid in range(len(lines)):
        strs.append(lines[lid].rstrip())

    return strs

# sort files in a folder by id value
def sortFilesByIdData(files):
    ids=[]
    for file in files:
        ids.append(int(file[file.find('_')+1: file.find('.txt')]))
    
    ids, filenames = zip(*sorted(zip(ids, files)))
    return filenames

# read sparse vector data to int strings
def readAllSparseStr(path, max_qty = None): 
    # read all files in the folder and sort them
    files=os.listdir(path)
    files=sortFilesByIdData(files)
    
    vector_str=[]
    for file in files:
        vector_str+=readEncodeFile(path+file)
    return vector_str

# creating index using vectors in string data
def createIndex(data_vector_str, M, efC, num_threads, post):
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : post}

    # Intitialize the library, specify the space, the type of the data and add data points 
    # Note that in the GENERIC case, data points are passed as strings!

    # index = nmslib.init(method='hnsw', space='bit_jaccard', data_type=nmslib.DataType.OBJECT_AS_STRING) 
    index = nmslib.init(method='hnsw', space='jaccard_sparse', data_type=nmslib.DataType.OBJECT_AS_STRING) 
    index.addDataPointBatch(data_vector_str) 

    # Create an index
    start = time.time()
    index.createIndex(index_time_params) 
    end = time.time() 
    print("Index-time parameters [{}] Indexing time: {}".format(index_time_params, (end-start)))
    print("{} vectors indexed".format(len(data_vector_str)))

    return index

def saveIndex(data_vector_str, M_list, efC_list, num_threads, post, path):
    for M in M_list:
        for efC in efC_list:
            # create index
            index=createIndex(data_vector_str, M, efC, num_threads, post)
            # Save a meta index and data
            fname=path+"index_"+str(M)+"-"+str(efC)+"-"+str(post)+".bin"
            index.saveIndex(fname, save_data=True)
            print("{} saved.".format(fname))


# generate indexes for all water bodies datasets
def generateIndexes(nodeCapacityPercList, dataFolder, folder, num_threads):
    print("Constructing and saving indexes.\ndata folder={} num_threads={}".format(dataFolder, num_threads))
    for i in range(len(nodeCapacityPercList)):
        print("nodeCapacityPerc={}".format(dataFolder, nodeCapacityPercList[i]))
        # read water nodies file data
        all_vector_str=readAllSparseStr(dataFolder+str(nodeCapacityPercList[i])+"/")

        start=0
        end=len(all_vector_str)
        data_percet=0.8
        data_start=start
        data_end=math.ceil((end-start)*data_percet)
        query_start=data_end
        query_end=end

        data_vector_str=all_vector_str[data_start:data_end]
        print("{} vectors found in data!".format(len(data_vector_str)))
        # query_vector_str=all_vector_str[query_start:query_end]
        # print("{} vectors found in query!".format(len(query_vector_str)))

        # Index parameters
        # ------------------------------------------------------------------
        # How many neighbors for a given node at non-zero layers (5-100)
        # At non-zero layer (bottom most) ther are 2*M neighbors (maxM0 variable to change default)
        # M = 50
        M_list=[20]
        # maxM0 =2**20

        # efC = 500            #ef construction (100-2000)
        # efC = 200            #ef construction (100-2000)
        efC_list=[200]
        post=1     #will do extra processing with [1, 2] values. No extra processing with 0

        # generate multiple indexes and save them
        path=folder+str(nodeCapacityPercList[i])+"/"   
        if not os.path.exists(path):
            os.makedirs(path)  
             
        saveIndex(data_vector_str, M_list, efC_list, num_threads, post, path)
        
# generate indexes for datasets for multiple threshold values
def generateIndexesth(nodeCapacityPercList, threshholds, dataFolder, folder, num_threads):
    print("Constructing and saving indexes.\ndata folder={} num_threads={}".format(dataFolder, num_threads))
    i=0
    for j in range(len(threshholds)):
        print("{} threshold={} nodeCapacityPerc={}".format(dataFolder, threshholds[j], nodeCapacityPercList[i]))
        # read water nodies file data
        all_vector_str=readAllSparseStr(dataFolder+str(threshholds[j])+"_"+str(nodeCapacityPercList[i])+"/")

        start=0
        end=len(all_vector_str)
        data_percet=0.8
        data_start=start
        data_end=math.ceil((end-start)*data_percet)
        query_start=data_end
        query_end=end

        data_vector_str=all_vector_str[data_start:data_end]
        print("{} vectors found in data!".format(len(data_vector_str)))
        # query_vector_str=all_vector_str[query_start:query_end]
        # print("{} vectors found in query!".format(len(query_vector_str)))

        # Index parameters
        # ------------------------------------------------------------------
        # How many neighbors for a given node at non-zero layers (5-100)
        # At non-zero layer (bottom most) ther are 2*M neighbors (maxM0 variable to change default)
        # M = 50
        M_list=[20]
        # maxM0 =2**20

        # efC = 500            #ef construction (100-2000)
        # efC = 200            #ef construction (100-2000)
        efC_list=[200]
        post=1     #will do extra processing with [1, 2] values. No extra processing with 0

        # generate multiple indexes and save them
        path=folder+str(threshholds[j])+"_"+str(nodeCapacityPercList[i])+"/"   
        if not os.path.exists(path):
            os.makedirs(path)  
             
        saveIndex(data_vector_str, M_list, efC_list, num_threads, post, path)

def index_parks(data_file, result_dir):
    # for parks
    # nodeCapacityPercPkList=[0.001, 0.002, 0.006, 0.012, 0.0015] # has all
    # nodeCapacityPercPkList=[0.012, 0.0015]
    num_threads = 240         # numbers of threads used for training
    # folder="/raid/ssEncodingData/indexes/pk_qtree_"
    # dataFolder="../encoding/pk"

    # generateIndexes(nodeCapacityPercPkList, threshholds, dataFolder, folder, num_threads)
    
    # threshold expermiment
    nodeCapacityPercPkList=[0.002] 
    threshholds=[0.25, 0.5, 0.75]
    dataFolder="/raid/ssEncodingData/encoding/pk"  
    folder="/raid/ssEncodingData/indexes/pk_qtree_th_"
    # generateIndexesth(nodeCapacityPercPkList, threshholds, dataFolder, folder, num_threads)
    generateIndexesth(nodeCapacityPercPkList, threshholds, data_file, result_dir, num_threads)

def index_water_bodies(data_file, result_dir):
    # for water bodies
    nodeCapacityPercWbList=[0.001, 0.003, 0.006, 0.0008]
    num_threads = 240         # numbers of threads used for training
    folder="/raid/ssEncodingData/indexes/wb_qtree_"
    dataFolder="../encoding/wb"

    # generateIndexes(nodeCapacityPercWbList, dataFolder, folder, num_threads)
    generateIndexes(nodeCapacityPercWbList, data_file, result_dir, num_threads)

def index_sports(data_file, result_dir):
    # for sports
    # nodeCapacityPercWbList=[0.001, 0.002, 0.0003, 0.0007]
    nodeCapacityPercWbList=[0.0056]
    num_threads = 64         # numbers of threads used for training
    folder="/raid/ssEncodingData/indexes/sp50k_qtree_"
    dataFolder="/raid/ssEncodingData/encoding/sports-50k"

    # generateIndexes(nodeCapacityPercWbList, dataFolder, folder, num_threads)
    generateIndexes(nodeCapacityPercWbList, data_file, result_dir, num_threads)

# def main():
#     # sports()
#     # water_bodies()
#     parks()

# if __name__=="__main__":
#     main()