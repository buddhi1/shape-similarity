# This file is used to generate ground truth for polygon similarity using Jaccard distance
# An are filtering is used to optimize the computation
# The results should exeed a prefined thrhold similarity value to optimize I/O

# Author: Buddhi Ashan M. K.
# Date: 01-19-2024

# creating the environment: conda create --name fast-mpi4py python=3.8 -y
# conda environment: conda activate fast-mpi4py
# Run with no hang up: nohup time python groundtruthWriting.py > groundtruthLog/out-pk-50k.log&

# import library
import numpy as np
from shapely.geometry.polygon import Polygon
from shapely import affinity
import matplotlib.pyplot as plt
import shapely.wkt
import math
import time 


import pandas as pd
# execute tasks in parallel in a for loop
from multiprocessing import Process, Array
from threading import Thread

# to set the path
import sys
sys.path.insert(1, '../lib/')
import wkthelper

wktList=[]

# center a given polygon
def centerPolygon(poly):
    # polygon translate offsets
    translate_poly = [poly.centroid.x*-1, poly.centroid.y*-1]
    # translate polygon to center at the origin
    return affinity.translate(poly, translate_poly[0], translate_poly[1])

def centerAllPolygons(wkts):
    outWkts=[]
    for i in range(len(wkts)):
        outWkts.append(centerPolygon(wkts[i]))
    return outWkts

# Functions to compare polygons without using area filter. Jaccard similarity is computed only for filtered candidates

# compare two given polygons using Jaccard distance 
def comparePolygonsFiltered(cBPol, cOPol, bPolArea, th, areaTh):
    oPolArea=cOPol.area
    diffArea=abs(bPolArea-oPolArea)
    maxArea=min([bPolArea, oPolArea])

    if diffArea/maxArea>areaTh :
        return -1
    else:
        intersectArea=cBPol.intersection(cOPol).area
        # unionArea=cBPol.union(cOPol).area
        unionArea=bPolArea+oPolArea-intersectArea

        # bToIntersectAreaPercentage=intersectArea/bPolArea  # intersect_area/base_area
        bToIntersectAreaPercentage=intersectArea/unionArea  # intersect_area/union_area
        if(bToIntersectAreaPercentage>th and intersectArea/oPolArea>th):
            # print("{} {}".format(oid, bToIntersectAreaPercentage))
            return bToIntersectAreaPercentage
        return -1


# find similar polygons for given polygon. Filtered by area
def FindSimilarPolygonsFiltered(pid, bid, cBPol, pCount, start, end, re, match, areaTh):
    localSize=math.floor((end-start)/pCount)
    s=start+(pid*localSize)
    e=start+localSize

    # floor function can add more workload for the last process
    if pid==pCount-1:
        e=end
    
    bPolArea=cBPol.area
    # print("{} {} {} {}".format(pid, localSize, start, end))
    for oid in range(s, e):
        if(bid!=oid):
            cOPol=wktList[oid]
            re[oid]=comparePolygonsFiltered(cBPol, cOPol, bPolArea, match, areaTh)
        else:
            re[oid]=1
    
    # sort is used here for timing purpose
    re.sort(reverse=True)

# prepare an array indicating the similar shapes in the dataset
# runs using multi-threads
# pCount: number of processors
# size: problems size
# bid: query polygon id
def shapeBasedBruteforceFilteredMultiProcess(bid, start, end, match, pCount, areaTh):
    results=Array("d", range(end-start))
    cBPol=wktList[bid]

    # create all tasks
    # processes = [Process(target=FindSimilarPolygons, args=(i, bid, cBPol, pCount, size, results, match)) for i in range(pCount)]
    processes = [Process(target=FindSimilarPolygonsFiltered, args=(i, bid, cBPol, pCount, start, end, results, match, areaTh)) for i in range(pCount)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    # print(flush=True)
    return results

def shapeBasedBruteforceFilteredMultiThread(bid, start, end, match, tCount, areaTh):
    results=Array("d", range(end-start))
    cBPol=wktList[bid]

    # create all tasks
    # threads = [Thread(target=FindSimilarPolygons, args=(i, bid, cBPol, pCount, size, results, match)) for i in range(pCount)]
    threads = [Thread(target=FindSimilarPolygonsFiltered, args=(i, bid, cBPol, tCount, start, end, results, match, areaTh)) for i in range(tCount)]
    # start all processes
    for thread in threads:
        thread.start()
    # wait for all processes to complete
    for thread in threads:
        thread.join()
    # report that all tasks are completed
    # print(flush=True)
    return results

# writing map to file. Each thread will write the results to its own file
def writeSimilarityMapToFileParalell(map, pstart, start, end, filename):
    filename+="_"+str(start)+"-"+str(end-1)
    f=open(filename, 'w+')

    bid=0
    # print("start={} end={}".format(0, len(map)))

    for id in range(len(map)):
        # ------ writing start -------
        f.write("%d" % (bid+pstart))
        for mid in range(len(map[id])):
            f.write(", %d" % map[id][mid][0])
        f.write("\n")
        # ------ Writing end ----------
        # print("Writing complete for bid:{}\n".format(bid))
        bid+=1
    f.close()
    # print("Writing to {} completed!".format(filename))

# create similar shape map. Multi processing
# Uses area filtered version of matching
def createSimilarityMapMultiProcess(pid, filename, fileSize, pCount, tCount, data_start, data_end, query_start, query_end, match, areaTh):
    map=[]
    localSize=math.floor((query_end-query_start)/pCount)
    s=query_start+(pid*localSize)
    e=s+localSize

    if(pid==pCount-1):
        e=query_end

    # print("pid={} start={} end={}".format(pid, s, e))

    filebidstart=s
    ls=s
    le=e
    count=1
    # for bid in range(s, e):
    # print("Similarity search on bid:{} ..".format(bid))
    similarityRow=[]
    similarity=shapeBasedBruteforceFilteredMultiThread(bid, data_start, data_end, match, tCount, areaTh)
    # similarity=shapeBasedBruteforceMultiThread(bid, len(wktList), match, tCount)

    for i in range(len(similarity)):
        if bid!=i and similarity[i]!=-1 and similarity[i]>0 and similarity[i]<=1:
            similarityRow.append([i, similarity[i]])
    # print("{} similar shapes found!\nAppending to map..".format(len(similarityRow)))

    similarityRow.sort(key=lambda x:x[1], reverse=True)
    map.append(similarityRow)
        # i=0
        # for score in similarityRow:
        #     print(score[0])
        #     map[pid*(data_end-data_start)+i]=score[0]
        #     i+=1
        
        # print("Search complete for bid:{}\n".format(bid))
        # if (s!=bid and (count)%fileSize==0) or bid==e-1:
        #     le=bid+1
        #     # print("pid={} ls={} le={} filebidstart={}".format(pid, ls, le, filebidstart))
        #     writeSimilarityMapToFileParalell(map, filebidstart, ls, le, filename)
        #     map=[]
        #     filebidstart=le
        #     ls=le
        # count+=1
    print("Process {} finished!".format(pid))
    

def createAllSimilarityMaps(filename, fileSize, pCount, tCount, data_start, data_end, query_start, query_end, match, areaTh):
    if pCount>(query_end-query_start):
        print("Multiprocesses reduced from {} to {}".format(pCount, query_end-query_start))
        pCount=query_end-query_start
    # results=Array("d", range((query_end-query_start)*(data_end-data_start)))

    # create all tasks
    processes = [Process(target=createSimilarityMapMultiProcess, args=(i, filename, fileSize, pCount, tCount, data_start, data_end, query_start, query_end, match, areaTh)) for i in range(pCount)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()
    # report that all tasks are completed
    # print(flush=True)
    # return results

def sports():
    # read and center the polygons
    print("Data reading")
    # wktAll = wkthelper.readWKTToList("../polygonalData/sports", poly_count=2000)
    wktAll = wkthelper.readWKTToList("../polygonalData/sports")
    # Center all polygons to origin
    print("Polygon centering started")
    global wktList
    wktList=centerAllPolygons(wktAll)
    print("Polygon centering complete!")
    
    filename="../warehouse/sports-query-16434/similarityMap"
    fileSize=400   # define max entries in a single file
    pCount=64
    tCount=2
    match=0.6
    areaTh=10 # if the base polygon and candidate polygon area difference is more than threhold, we may filter them out
    print("Data writing to {}".format(filename))
    print("#processes={} #threads={} similarity matching threhold={} area filter threhold={}".format(pCount, tCount, match, areaTh))
    data_start=0
    data_end=math.ceil(len(wktList)*0.8)
    # data_end=1000
    # query_start=data_end
    query_start=1643448
    query_end=query_start+50000
    print("Data from {} to {}. Query from {} to {}.".format(data_start, data_end, query_start, query_end))
    
    results=[0]*(data_end-data_start)
    start = time.time()
    FindSimilarPolygonsFiltered(0, query_start, wktList[query_start], 1, data_start, data_end, results, match, areaTh)
    end = time.time()

    print("Latency= {} (Seconds)".format(end-start))

    # createAllSimilarityMaps(filename, fileSize, pCount, tCount, data_start, data_end, query_start, query_end, match, areaTh)

def water_bodies():
    # read and center the polygons
    print("Data reading")
    # wktAll = wkthelper.readWaterBodies("../polygonalData/water_bodies.txt", poly_count=2000)
    wktAll = wkthelper.readWaterBodies("../polygonalData/water_bodies.txt")
    
    # Center all polygons to origin
    print("Polygon centering started")
    global wktList
    wktList=centerAllPolygons(wktAll)
    print("Polygon centering complete!")
    
    filename="../warehouse/wb-query-35884/similarityMap"
    fileSize=400   # define max entries in a single file
    pCount=60
    tCount=2
    match=0.6
    areaTh=10 # if the base polygon and candidate polygon area difference is more than threhold, we may filter them out
    print("Data writing to {}".format(filename))
    print("#processes={} #threads={} similarity matching threhold={} area filter threhold={}".format(pCount, tCount, match, areaTh))
    data_start=0
    data_end=math.ceil(len(wktList)*0.8)
    
    # data_end=1000
    # query_start=data_end
    # query_end=query_start+20

    query_start=358840
    query_end=query_start+89710
    print("Data from {} to {}. Query from {} to {}.".format(data_start, data_end, query_start, query_end))
    
    results=[0]*(data_end-data_start)
    start = time.time()
    FindSimilarPolygonsFiltered(0, query_start, wktList[query_start], 1, data_start, data_end, results, match, areaTh)
    end = time.time()

    print("Latency= {} (Seconds)".format(end-start))

    # createAllSimilarityMaps(filename, fileSize, pCount, tCount, data_start, data_end, query_start, query_end, match, areaTh)

def parks():
    # read and center the polygons
    print("Data reading")
    wktAll = wkthelper.readParks("../polygonalData/parks.tsv")
    # wktAll = wkthelper.readParks("../polygonalData/parks.tsv", poly_count=50)
    
    # Center all polygons to origin
    print("Polygon centering started")
    global wktList
    wktList=centerAllPolygons(wktAll)
    print("Polygon centering complete!")
    
    filename="/raid/ssEncodingData/warehouse/pk-query-50k/similarityMap"
    fileSize=400   # define max entries in a single file
    pCount=60
    tCount=2
    match=0.6
    areaTh=10 # if the base polygon and candidate polygon area difference is more than threhold, we may filter them out
    print("Data writing to {}".format(filename))
    print("#processes={} #threads={} similarity matching threhold={} area filter threhold={}".format(pCount, tCount, match, areaTh))
    data_start=0
    data_end=math.ceil(len(wktList)*0.8)
    
    # data_end=1000
    # query_start=data_end
    # query_end=query_start+20

    query_start=data_end
    query_end=len(wktList)
    print("Data from {} to {}. Query from {} to {}.".format(data_start, data_end, query_start, query_end))
    
    results=[0]*(data_end-data_start)
    start = time.time()
    FindSimilarPolygonsFiltered(0, query_start, wktList[query_start], 1, data_start, data_end, results, match, areaTh)
    end = time.time()

    print("Latency= {} (Seconds)".format(end-start))
    # createAllSimilarityMaps(filename, fileSize, pCount, tCount, data_start, data_end, query_start, query_end, match, areaTh)


def main():
    # parks()
    # sports()
    water_bodies()

if __name__=="__main__":
    main()




# Data reading
# 233773 polygons found
# Polygon centering started
# Polygon centering complete!
# Data writing to /raid/ssEncodingData/warehouse/pk-query-50k/similarityMap
# #processes=60 #threads=2 similarity matching threhold=0.6 area filter threhold=10
# Data from 0 to 187019. Query from 187019 to 233773.
# Latency= 9.3398916721344 (Seconds)


# (ssearch2) buddhi@dgxa100:~/ssEncoding/src$ python groundtruthTiming.py 
# Data reading
# 1753989 polygons found
# Polygon centering started
# Polygon centering complete!
# Data writing to ../warehouse/sports-query-16434/similarityMap
# #processes=64 #threads=2 similarity matching threhold=0.6 area filter threhold=10
# Data from 0 to 1403192. Query from 1643448 to 1693448.
# Latency= 42.7561514377594 (Seconds)



# (ssearch2) buddhi@dgxa100:~/ssEncoding/src$ python groundtruthTiming.py 
# Data reading
# 448550 polygons found
# Polygon centering started
# Polygon centering complete!
# Data writing to ../warehouse/wb-query-35884/similarityMap
# #processes=60 #threads=2 similarity matching threhold=0.6 area filter threhold=10
# Data from 0 to 358840. Query from 358840 to 448550.
# Latency= 21.96523690223694 (Seconds)