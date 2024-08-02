import shapely.wkt
from shapely.geometry.polygon import Polygon
from shapely import affinity
import math
import matplotlib.pyplot as plt
# from src.common_imports import shapely, Polygon, affinity, math, plt

# Reading from a file
def readWKTToList(fileName, start=0, line_count=-1, poly_count=-1):
    file1 = open(fileName, "r")
    inputWKTs = []
    i = 0
    while True:
        # read line
        line = file1.readline()
        # if line is empty end of file is reached
        if (line_count > 0 and line_count <= i) or (poly_count > 0 and poly_count <= len(inputWKTs)) or not line:
            break
        # print("line: {}".format(line))
        # remove row number.
        line = line[line.find('\t'):]

        # meta_data=line[line.find('[')+1:line.find(']')]
        # if (meta_data.find("lake")<0):
        #     continue

        # convert to wkt object
        wktStr = shapely.wkt.loads(line)
        # load only polygons
        # print(wktStr.geom_type)
        # check inside geometry collections
        # if wktStr.geom_type == "GeometryCollection":
        #     for i in range(len(wktStr.geoms)):
        #         # only load polygons without holes
        #         if wktStr.geoms[i].geom_type == "Polygon" and len(wktStr.geoms[i].interiors) <= 0 and len(wktStr.geoms[i].exterior.coords) > 0:
        #             if wktStr.geoms[i].is_valid:
        #                 inputWKTs.append(wktStr.geoms[i])
        #                 i+=1

        # only load polygons without holes
        if wktStr.geom_type == "Polygon" and len(wktStr.interiors) <= 0 and len(wktStr.exterior.coords) > 0:
            if wktStr.is_valid:
                if(start <= i):
                    inputWKTs.append(wktStr)
                i += 1
    file1.close()
    print("{} polygons found".format(len(inputWKTs)))
    return inputWKTs

# read water_bodies.txt file
def readWaterBodies(fileName, start=0, line_count=-1, poly_count=-1):
    file1=open(fileName, "r")
    inputWKTs=[]
    i=0
    while True:
        # read line
        line=file1.readline()
        # if line is empty end of file is reached
        if (line_count>0 and line_count<=i) or (poly_count>0 and poly_count<=len(inputWKTs)) or not line:
            break

        # remove row number.
        line=line[line.find(' ')+1:]


        # print(line)
        vertices=line.split(' ')
        vl=' '.join(vertices).split()
        vertices=list(map(float, vl))
        # print(vertices)

        coord=[]
        for vid in range(4, len(vertices), 2):   # skips the first 4 entries which defines the MBR of the polygon
            coord.append((vertices[vid], vertices[vid+1]))

        # convert list of coordinates to wkt polygon
        wktStr=Polygon(coord)

        # only load polygons without holes
        if len(wktStr.interiors)<=0 and len(wktStr.exterior.coords)>0:
            if wktStr.is_valid:
                if(start<=i):
                    inputWKTs.append(wktStr)
                i+=1
    file1.close()
    print("{} polygons found".format(len(inputWKTs)))
    return inputWKTs

# Reading parks.tsv file
def readParks(fileName, start=0, line_count=-1, poly_count=-1):
    file1 = open(fileName, "r")
    inputWKTs = []
    i = 0
    while True:
        # read line
        line = file1.readline()

        # if line is empty end of file is reached
        if (line_count > 0 and line_count <= i) or (poly_count > 0 and poly_count <= len(inputWKTs)) or not line:
            break
        # print("line: {}".format(line))
        # remove row number.
        line = line[line.find('\t')+1:]

        # meta_data=line[line.find('[')+1:line.find(']')]
        # if (meta_data.find("lake")<0):
        #     continue

        # convert to wkt object
        ls=shapely.wkt.loads(line)
        if len(ls.coords)>2:  # avoids any points or lines
            wktStr = Polygon(ls.coords)

            # only load polygons without holes
            if wktStr.geom_type == "Polygon" and len(wktStr.interiors) <= 0 and len(wktStr.exterior.coords) > 0:
                if wktStr.is_valid:
                    if(start <= i):
                        inputWKTs.append(wktStr)
                    i += 1
    file1.close()
    print("{} polygons found".format(len(inputWKTs)))
    return inputWKTs

# return the list of area for the given wkt list
def polyAreaArray(wkts, start, end):
    poly_area = []
    for i in range(start, end):
        poly_area.append(wkts[i].area)
    print("Area range: {} - {}".format(min(poly_area), max(poly_area)))
    return poly_area

# plot given area list as a histogram
def plotAreaHist(poly_area, islog=True):
    if islog:
        plt.yscale("log")
    plt.hist(poly_area)
    plt.show

# plot given area list as a bar chart
def plotAreaBarChart(poly_area, issort=True, islog=True):    
    if issort:
        poly_area.sort()
    if islog:
        plt.yscale("log")
    plt.bar([x for x in range(len(poly_area))], poly_area)
    # Labeling the X-axis
    plt.xlabel("Polygons")

    # Labeling the Y-axis
    plt.ylabel("Area (m\u00B2)")
    plt.show

# return min area of the given polygon list
def minAreaWkt(wktList):
    area_min=wktList[0].area
    for i in range(len(wktList)):
        area=wktList[i].area
        if area_min>area:
            area_min=area
    print("Min Area={}".format(area_min))

# return average area of the given polyogn list
def wktAvgArea(wktList):
    area_sum=0
    for i in range(len(wktList)):
        area=wktList[i].area
        area_sum+=area
    area_avg=area_sum/len(wktList)
    print("Avg Area={}".format(area_avg))

# normalize polygons by a given area range
def normalizePolygonsByAreaRange(wkts, wkt_area_list, base_area, max_area, start, end):
    mid_area = base_area+((max_area-base_area)/2)
    # print(mid_area)
    normalized_wkts = []
    for i in range(start, end):
        if wkt_area_list[i] > max_area or wkt_area_list[i] < base_area:
            scale = math.sqrt(mid_area/(wkt_area_list[i]))
            # print("i={}: {} mid: {} area: {}".format(i, scale, mid_area, wkt_area_list[i]))
            normalized_wkts.append(affinity.scale(wkts[i], xfact=scale, yfact=scale))
        else:
            normalized_wkts.append(wkts[i])
    return normalized_wkts

# normalize polygons by a given area
def normalizePolygonsByAreaVal(wkts, wkt_area_list, mid_area, start, end):
    normalized_wkts = []
    for i in range(start, end):
        scale_factor = math.sqrt(mid_area/(wkt_area_list[i]))
        if(scale_factor!=1):
            # print("i={}: {} mid: {} area: {}".format(i, scale, mid_area, wkt_area_list[i]))
            normalized_wkts.append(affinity.scale(wkts[i], xfact=scale_factor, yfact=scale_factor))
    return normalized_wkts
