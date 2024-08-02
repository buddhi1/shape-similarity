# converting polygons in WKT format to sparse matrices in parallel using a quad tree-based non-uniform grid

# Author: Buddhi Ashan M. K.
# Date: 01-31-2024

# conda environment: conda activate ssearch2
# Run with no hang up: nohup time python encodeFileWritingQuadTree.py > logs/encodingQuadTree/eout-pk50k-0.00026.log&

import numpy as np
from shapely.geometry.polygon import Polygon
from shapely import affinity
import matplotlib.pyplot as plt
import shapely.wkt
import math
from shapely.strtree import STRtree

# execute tasks in parallel in a for loop
from multiprocessing import Process

import os
# # to set the path
import sys
# sys.path.insert(1, '../utils/')
# import wkthelper
# from quadtree import quadtree

# Relative import
from src.utils import wkthelper
from src.utils.quadtree import quadtree


# claculate the MBR which can contain all of the polygons in the set.
# It can be boundaries of multiple polygons or a MBR of a single polygon
def findSetMBR(inputWKTs, end, start=0):
    # mbr: minX, minY, maxX, maxY
    mbr=list(inputWKTs[start].bounds)
    for i in range(start+1, end):
        poly_mbr=list(inputWKTs[i].bounds)
        if(poly_mbr[0]<mbr[0]):
            mbr[0]=poly_mbr[0]
        if(poly_mbr[1]<mbr[1]):
            mbr[1]=poly_mbr[1]
        if(poly_mbr[2]>mbr[2]):
            mbr[2]=poly_mbr[2]
        if(poly_mbr[3]>mbr[3]):
            mbr[3]=poly_mbr[3]
    print("Global MBR of [{}-{}] = {}".format(start, end, mbr))
    return mbr

def initialPolygonCentering(inputWkts, end, start=0):
    centered_wkts=[]
    for i in range(start, end):
        c=inputWkts[i].centroid
        centered_wkts.append(affinity.translate(inputWkts[i], c.centroid.x*-1, c.centroid.y*-1))
    print("{} polygons centered to origin".format(len(centered_wkts)))

    return centered_wkts

# read wkt data, center polygons, and construct the quad tree  
def preProcessQuadTree(wktList, data_start, data_end, data_percet, nodeCapacityPerc):
    node_capacity=math.ceil((data_end-data_start)*nodeCapacityPerc)
    # find MBR that contains all polygons in the set
    global_mbr=findSetMBR(wktList, start=0, end=len(wktList))
    qt=quadtree(global_mbr, node_capacity)
    for i in range(data_start, data_end):
        xx, yy = wktList[i].exterior.xy
        for x,y in zip(xx, yy):
            qt.insert(x, y, f"Point({x}, {y})")
    
    # get all bounding boxes
    qt.get_all_bounding_boxes()
    # qt.plot_quadtree()
    print("Quad tree of size {} and height {} generated using {} polygons".format(len(qt.bounding_boxes), qt.levels, data_end-data_start))
    return wktList, qt, global_mbr

# writing map to file. Each thread will write the results to its own file
def writeEncodeStrToFile(filename, vectors, start):
    filename+="_"+str(start)+".txt"
    f=open(filename, 'w+')

    for id in range(len(vectors)):
        row=vectors[id]
        # ------ writing start -------
        for rid in range(len(row)):
            line=str(row[rid])+" "
            f.write(line)
        f.write("\n")
        # ------ Writing end ----------
    f.close()

# produce feature vector for each polygon using the quad tree grid
# R-tree based method. R tree is create for cells and then quaried for each polygon
# output: string of activated cell ids
# using Jaccard similarity threhold in each cell
def encodePolygonsRtreeJaccardStr(filename, fileSize, rtree, qt, wktList, start, end, pCount, pid, th_area, foreground, weighted):
    qtree_boxes=qt.bounding_boxes
    qtree_box_levels=qt.bounding_box_levels
    levels=qt.levels

    localSize=math.floor((end-start)/pCount)
    if (end-start)<pCount:
        localSize=1

    s=start+pid*localSize
    e=s+localSize
    if pid==pCount-1:
        e=end
    csize=len(qtree_boxes)

    vectors=[]
    
    count=1
    ls=s
    le=e
    for j in range(s, e):
        # find candidate cells by querying the rtree
        candidates=rtree.query(wktList[j])
        candidates.sort()
        row=(j-start)*csize
        # print("j={} j-start={} row={}".format(j, (j-start), row))

        row=[]
        for candidate in candidates:
            box=qtree_boxes[candidate]
            box_level=qtree_box_levels[candidate]
            box_geom=Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
            # print("i {} Wkt: {}".format(j, wktList[j]))
            if box_geom.intersects(wktList[j]):
                intersectArea=box_geom.intersection(wktList[j]).area
                box_area=box_geom.area

                if intersectArea/box_area>=th_area*(box_level/levels):   # change this to Jaccrd distance
                    row.append(candidate)
            
        vectors.append(row)

        if (s!=j and (count)%fileSize==0) or j==e-1:
            le=j+1
            # writing to file
            writeEncodeStrToFile(filename, vectors, ls)
            vectors=[]
            filebidstart=le
            ls=le
        count+=1
    print("Process {} finished.".format(pid))

# produce feature vector for each polygon using the quad tree grid
# R-tree based method. R tree is create for cells and then quaried for each polygon
# output: string of activated cell ids
# Only checks for cell intersection with polygon
def encodePolygonsRtreeStr(filename, fileSize, rtree, qt, wktList, start, end, pCount, pid, th_area, foreground, weighted):
    qtree_boxes=qt.bounding_boxes
    qtree_box_levels=qt.bounding_box_levels
    levels=qt.levels

    localSize=math.floor((end-start)/pCount)
    if (end-start)<pCount:
        localSize=1

    s=start+pid*localSize
    e=s+localSize
    if pid==pCount-1:
        e=end
    csize=len(qtree_boxes)

    vectors=[]
    
    count=1
    ls=s
    le=e
    for j in range(s, e):
        # find candidate cells by querying the rtree
        candidates=rtree.query(wktList[j])
        candidates.sort()
        row=(j-start)*csize
        # print("j={} j-start={} row={}".format(j, (j-start), row))

        row=[]
        for candidate in candidates:
            box=qtree_boxes[candidate]
            box_level=qtree_box_levels[candidate]
            box_geom=Polygon([(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])])
            box_geom_area=box_geom.area
            # print("i {} Wkt: {}".format(j, wktList[j]))
            if box_geom.intersects(wktList[j]):
                row.append(candidate)
                # intersectArea = box_geom.intersection(wktList[j]).area
                # if intersectArea/box_geom_area >= th_area: 
                #     row.append(candidate)
            
        vectors.append(row)

        if (s!=j and (count)%fileSize==0) or j==e-1:
            le=j+1
            # writing to file
            writeEncodeStrToFile(filename, vectors, ls)
            vectors=[]
            filebidstart=le
            ls=le
        count+=1
    print("Process {} finished.".format(pid))

# multi-process encoding
def multiProcessEncodingWriting(filename, fileSize, rtree, qt, wktList, start, end, arg, foreground, weighted, pCount):

    # adjust pCount if data size is too small
    if end-start<pCount:
        pCount=end-start
    
    if pCount>(end-start):
        print("Multiprocesses reduced from {} to {}".format(pCount, end-start))
        pCount=end-start

    # create all tasks
    processes=[Process(target=encodePolygonsRtreeStr, args=(filename, fileSize, rtree, qt, wktList, start, end, pCount, i, arg, foreground, weighted)) for i in range(pCount)]
    # start all processes
    for process in processes:
        process.start()
    # wait for all processes to complete
    for process in processes:
        process.join()

# create sparse matrix of vector polygons
def writeWKTListToSparseMatrix(filename, fileSize, wktList, data_start, data_end, query_start, query_end, data_percet, nodeCapacityPerc, th_area, pCount):   
    wktList,qt,global_mbr=preProcessQuadTree(wktList, data_start, data_end, data_percet, nodeCapacityPerc)

    # print("Size {}".format(len(wktList)))
    
    # convert quad tree nodes into a list of polygons
    wktListQtBoxes=qt.convertQTToPolys()
    # buid rtee using cells of the quadtree. Cells are in the form of polygons
    rtree=STRtree(wktListQtBoxes)

    multiProcessEncodingWriting(filename, fileSize, rtree, qt, wktList, data_start, query_end, th_area, 1, True, pCount)
    # print("embed     weighted: {}".format(queryVectors))

def encode_quad_tree_sports(data_file, result_dir):
    # print("quad sports")
    # dataseze=1753989
    # dataFile="../polygonalData/sports"
    wktList = wkthelper.readWKTToList(data_file)
    # wktList = wkthelper.readWKTToList(dataFile, poly_count=50000)
    # center all polygons
    wktList=initialPolygonCentering(wktList, end=len(wktList), start=0)


    data_percet=0.8
    th_area=0.75
    nodeCapacityPerc=0.0056

    data_start=0
    data_end=math.ceil(len(wktList)*data_percet)
    query_start=data_end
    query_end=len(wktList)

    # Define the number of threads to use
    pCount=64
    fileSize=1000


    # folder_name = "/raid/ssEncodingData/encoding/sports-50k"+str(nodeCapacityPerc)
    folder_name = result_dir+"/sports-50k"+str(nodeCapacityPerc)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename=folder_name+"/weightint"

    print("Data: [{} to {}]. Query: [{} to {}]".format(data_start, data_end, query_start, query_end))
    print("Saving to {}. Max points in a quad tree cell {}".format(folder_name, (len(wktList))*nodeCapacityPerc))

    writeWKTListToSparseMatrix(filename, fileSize, wktList, data_start, data_end, query_start, query_end, data_percet, nodeCapacityPerc, th_area, pCount)

def encode_quad_tree_water_bodies(data_file, result_dir):
    # print("quad wt")
    # datasiez=448550
    # wktAll = wkthelper.readWaterBodies("../polygonalData/water_bodies.txt")
    wktAll = wkthelper.readWaterBodies(data_file)
    # wktAll = wkthelper.readWaterBodies("../polygonalData/water_bodies.txt", poly_count=100)
    # center all polygons
    wktList=initialPolygonCentering(wktAll, end=len(wktAll), start=0)


    data_percet=0.8
    th_area=0.75
    nodeCapacityPerc=0.0008

    data_start=0
    data_end=math.ceil(len(wktList)*data_percet)
    query_start=data_end
    query_end=len(wktList)

    # Define the number of threads to use
    pCount=240
    fileSize=1000


    # folder_name = "../encoding/wb"+str(nodeCapacityPerc)
    folder_name = result_dir+"/wb"+str(nodeCapacityPerc)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename=folder_name+"/weightint"

    print("Data: [{} to {}]. Query: [{} to {}]".format(data_start, data_end, query_start, query_end))
    print("Saving to {}. Max points in a quad tree cell {}".format(folder_name, (len(wktList))*nodeCapacityPerc))

    writeWKTListToSparseMatrix(filename, fileSize, wktList, data_start, data_end, query_start, query_end, data_percet, nodeCapacityPerc, th_area, pCount)

def encode_quad_tree_parks(data_file, result_dir):
    # print("quad parks")
    # datasiez=233773
    # wktAll = wkthelper.readParks("../polygonalData/parks.tsv")
    # wktAll = wkthelper.readParks(data_file)
    wktAll = wkthelper.readParks(data_file, poly_count=50000)
    # center all polygons
    wktList=initialPolygonCentering(wktAll, end=len(wktAll), start=0)


    data_percet=0.8
    # th_area=0.75
    th_area=0
    # 0.002, 0.006, 0.012, 0.0015
    nodeCapacityPerc=0.00026
    # nodeCapacityPerc=0.2

    data_start=0
    data_end=math.ceil(len(wktList)*data_percet)
    query_start=data_end
    query_end=len(wktList)

    # Define the number of threads to use
    pCount=180
    fileSize=1000


    # folder_name = "/raid/ssEncodingData/encoding/pk"+str(th_area)+"_"+str(nodeCapacityPerc)
    # folder_name = "/raid/ssEncodingData/encoding/pk-50k"+str(nodeCapacityPerc)
    folder_name = result_dir+"/pk-50k"+str(nodeCapacityPerc)
    # folder_name = "../encoding/pk"+str(th_area)+"_"+str(nodeCapacityPerc)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    filename=folder_name+"/weightint"

    print("Data: [{} to {}]. Query: [{} to {}]".format(data_start, data_end, query_start, query_end))
    print("Saving to {}. Max points in a quad tree cell {}".format(folder_name, (len(wktList))*nodeCapacityPerc))

    writeWKTListToSparseMatrix(filename, fileSize, wktList, data_start, data_end, query_start, query_end, data_percet, nodeCapacityPerc, th_area, pCount)


# def main():
#     # sports()
#     # water_bodies()
#     # parks()

# if __name__=="__main__":
#     main()