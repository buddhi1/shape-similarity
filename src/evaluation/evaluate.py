# Bulk testing and reporting recall rates

# Author: Buddhi Ashan M. K.
# Date: 02-07-2024

# conda environment: conda activate ssearch2
# Run with no hang up: nohup time python bulkTesting.py > testingData/tout-pk-th_0.002.log&

# import library
import numpy as np
import geopandas
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
from scipy.sparse import csr_matrix 
from sklearn.model_selection import train_test_split 
print(sys.version)
print("NMSLIB version:", nmslib.__version__)

import os

from multiprocessing import Process, Array

# to set the path
import sys
# sys.path.insert(1, '../lib/')
# import wkthelper
# from quadtree import quadtree
# from grid import grid

from src.utils import wkthelper
from src.utils.quadtree import quadtree
from src.utils.grid import grid

def query(index, query_vector_str, K, efS, num_threads):
      # Setting query-time parameters
      query_time_params = {'efSearch': efS}
      print('Setting query-time parameters', query_time_params)
      index.setQueryTimeParams(query_time_params) 

      # Querying
      query_qty = len(query_vector_str)
      start = time.time() 
      nbrs = index.knnQueryBatch(query_vector_str, k = K, num_threads = num_threads)
      end = time.time() 
      print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
            (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty)) 
      return nbrs


# More convinient and latest method to read ground truth files without bach concept
# read a ground truth file to list
def readGroundTruthFile(filename):
    f=open(filename, 'r')
    lines=f.readlines()
    f.close()

    gt=[]
    for line in lines:
        dataRow=line.split(', ')
        if len(dataRow)<=1:
            gt.append([])
        else:
            gtRow=[]
            for data in dataRow:
                gtRow.append(int(data))
            gt.append(gtRow)
    return gt

# sort files in a folder by id value
def sortFilesById(files):
    ids=[]
    for file in files:
        ids.append(int(file[file.find('_')+1: file.find('-')]))
    
    ids, filenames = zip(*sorted(zip(ids, files)))
    return filenames

# read all files in the directory
def readAllGroundTruthFiles(path):
    files=os.listdir(path)
    files=sortFilesById(files)
    print(len(files))

    gtArray=[]
    for file in files:
        gtArray=gtArray+readGroundTruthFile(path+file)
    
    return gtArray

# filter out k similar values for each query polygon
def selectKGroundTruth(gs_all, k):
    gs=[]
    for i in range(len(gs_all)):
        gt=[]
        j=0
        while(j < len(gs_all[i]) and k > j):
            gt.append(gs_all[i][j])
            j+=1
        gs.append(gt)
    return gs

# compute recall when ground truth only has data for query data  
def computeRecallQueryOnly(gs, nbrs, query_qty):
    recall=0.0
    gs_nonzero_count=0

    for i in range(0, query_qty):
        if len(gs[i])>0:
            correct_set = set(gs[i])
            ret_set = set(nbrs[i][0])
            recall = recall + float(len(correct_set.intersection(ret_set))) / len(correct_set)
            gs_nonzero_count+=1
    recall_full = recall / query_qty
    recall = recall / gs_nonzero_count
    # print('Full kNN recall {}\nkNN recall {}'.format(recall_full, recall))
    return 'Full kNN recall {} kNN recall {}'.format(recall_full, recall)

# read sparse vectors into csr matrix

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
    a=str("10")
    
    vector_str=[]
    for file in files:
        vector_str+=readEncodeFile(path+file)
    return vector_str


def bulkTesting(gs, nodeCapacityPerc, query_vector_str, M_list, efS_list, K_list, num_threads, ifile, dataset, path):
    for M in M_list:
        for efS in efS_list:
            # Note again that in the GENERIC case, data points are passed as strings!
            index = nmslib.init(method='hnsw', space='jaccard_sparse', data_type=nmslib.DataType.OBJECT_AS_STRING) 
            fname=ifile+str(nodeCapacityPerc)+"/index_"+str(M)+"-"+str(efS)+"-1.bin"
            print(fname)

            # Re-load the index and the data
            index.loadIndex(fname, load_data=True)

            print(path+dataset+"_"+str(nodeCapacityPerc)+"-"+str(M)+"-"+str(efS))
            f = open(path+dataset+"_"+str(nodeCapacityPerc)+"-"+str(M)+"-"+str(efS), 'w')
            for K in K_list:
                # Query and recall computation
                # Query
                nbrs=query(index, query_vector_str, K, efS, num_threads)

                # select k ground truth
                gsk=selectKGroundTruth(gs, K)
                # Calculate recall rate
                msg=computeRecallQueryOnly(gsk, nbrs, query_qty=len(nbrs))
                print(msg)
                f.write("K="+str(K)+" "+msg+"\n")
            f.close

def testDataset(dataset, filename, ifile, efile, nodeCapacityPercList, num_threads):
    # read ground truth from file
    gs=readAllGroundTruthFiles(filename)
    # M_list=[20, 50]
    # efS_list=[200, 400]
    # K_list=[50, 500]
    M_list=[20]
    efS_list=[200]
    K_list=[50, 500]
    path="../testingData/"

    for nodeCapacityPerc in nodeCapacityPercList:
        # read parks file data
        all_vector_str=readAllSparseStr(efile+str(nodeCapacityPerc)+"/")

        start=0
        end=len(all_vector_str)
        data_percet=0.8
        data_start=start
        data_end=math.ceil((end-start)*data_percet)
        query_start=data_end
        query_end=end

        query_vector_str=all_vector_str[query_start:query_end]
        print("{} vectors found in query!".format(len(query_vector_str)))                                                                   

        bulkTesting(gs, nodeCapacityPerc, query_vector_str, M_list, efS_list, K_list, num_threads, ifile, dataset, path)

# multiple threshold experiment

def bulkTestingth(gs, nodeCapacityPerc, threshold, query_vector_str, M_list, efS_list, K_list, num_threads, ifile, dataset, path):
    for M in M_list:
        for efS in efS_list:
            # Note again that in the GENERIC case, data points are passed as strings!
            index = nmslib.init(method='hnsw', space='jaccard_sparse', data_type=nmslib.DataType.OBJECT_AS_STRING) 
            fname=ifile+str(threshold)+"_"+str(nodeCapacityPerc)+"/index_"+str(M)+"-"+str(efS)+"-1.bin"
            print(fname)

            # Re-load the index and the data
            index.loadIndex(fname, load_data=True)

            print(path+dataset+"_"+str(threshold)+"_"+str(nodeCapacityPerc)+"-"+str(M)+"-"+str(efS))
            f = open(path+dataset+"_"+str(threshold)+"_"+str(nodeCapacityPerc)+"-"+str(M)+"-"+str(efS), 'w')
            for K in K_list:
                # Query and recall computation
                # Query
                nbrs=query(index, query_vector_str, K, efS, num_threads)

                # select k ground truth
                gsk=selectKGroundTruth(gs, K)
                # Calculate recall rate
                msg=computeRecallQueryOnly(gsk, nbrs, query_qty=len(nbrs))
                print(msg)
                f.write("K="+str(K)+" "+msg+"\n")
            f.close

def testDatasetth(dataset, filename, ifile, efile, nodeCapacityPercList, thresholds, num_threads):
    # read ground truth from file
    gs=readAllGroundTruthFiles(filename)
    # M_list=[20, 50]
    # efS_list=[200, 400]
    # K_list=[50, 500]
    M_list=[20]
    efS_list=[200]
    K_list=[50, 500]
    path="../testingData/"

    i=0
    for j in range(len(thresholds)):
        # read parks file data
        all_vector_str=readAllSparseStr(efile+str(thresholds[j])+"_"+str(nodeCapacityPercList[i])+"/")

        start=0
        end=len(all_vector_str)
        data_percet=0.8
        data_start=start
        data_end=math.ceil((end-start)*data_percet)
        query_start=data_end
        query_end=end

        query_vector_str=all_vector_str[query_start:query_end]
        print("{} vectors found in query!".format(len(query_vector_str)))                                                                   

        bulkTestingth(gs, nodeCapacityPercList[i], thresholds[j], query_vector_str, M_list, efS_list, K_list, num_threads, ifile, dataset, path)


def evaluate_parksth():
    # print("evaluate pars th")
    
    filename="/raid/ssEncodingData/warehouse/pk-query-187019/"
    dataset="pk"
    ifile="/raid/ssEncodingData/indexes/pk_qtree_th_"
    efile="/raid/ssEncodingData/encoding/pk"

    nodeCapacityPercList=[0.002]
    thresholds=[0.25, 0.5, 0.75] 
    num_threads=210

    testDatasetth(dataset, filename, ifile, efile, nodeCapacityPercList, thresholds, num_threads)

def evaluate_parks(groundtruth_dir, index_dir, encoding_dir):
    # print("evaluate pars")

    filename="/raid/ssEncodingData/warehouse/pk-query-187019/"
    dataset="pk"
    ifile="/raid/ssEncodingData/indexes/pk_qtree_"
    efile="../encoding/pk"

    nodeCapacityPercList=[0.002, 0.006, 0.012, 0.0015] # all
    # nodeCapacityPercList=[0.002, 0.006] 
    num_threads=210

    # testDataset(dataset, filename, ifile, efile, nodeCapacityPercList, num_threads)
    testDataset(dataset, groundtruth_dir, index_dir, efile, nodeCapacityPercList, num_threads)

def evaluate_water_bodies(groundtruth_dir, index_dir, encoding_dir):
    # print("evaluate wt")

    filename="/raid/ssEncodingData/warehouse/wb-query-358840/"
    dataset="wb"
    ifile="/raid/ssEncodingData/indexes/wb_qtree_"
    efile="/raid/ssEncodingData/encoding/wb"

    nodeCapacityPercList=[0.006, 0.003]
    num_threads=210

    # testDataset(dataset, filename, ifile, efile, nodeCapacityPercList, num_threads)
    testDataset(dataset, groundtruth_dir, index_dir, encoding_dir, nodeCapacityPercList, num_threads)

def evaluate_sports(groundtruth_dir, index_dir, encoding_dir):
    # print("evaluate sporst")

    filename="/raid/ssEncodingData/warehouse/sports_all-query/"
    dataset="wb"
    ifile="/raid/ssEncodingData/indexes/sp_qtree_"
    efile="/raid/ssEncodingData/encoding/sports"

    # nodeCapacityPercList=[0.002, 0.0007]
    nodeCapacityPercList=[0.0007]
    num_threads=210

    # testDataset(dataset, filename, ifile, efile, nodeCapacityPercList, num_threads)
    testDataset(dataset, groundtruth_dir, index_dir, encoding_dir, nodeCapacityPercList, num_threads)


# def main():
#     # water_bodies()
#     # sports()
#     # parks()
#     parksth()

# if __name__=="__main__":
#     main()