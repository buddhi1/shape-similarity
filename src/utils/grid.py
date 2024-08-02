import numpy as np
from shapely import affinity
from shapely.geometry.polygon import Polygon
import matplotlib.pyplot as plt


# grid class
class grid:
    offsetFunction=[0, 0]

    def __init__(self, global_mbr, rowNum, colNum):
        self.rowNum = rowNum
        self.colNum = colNum
        self.setMbr(global_mbr)
        self.cellWidth=(self.width/self.rowNum) 
        self.cellHeight=(self.height/self.colNum)
        self.raster = np.zeros((self.rowNum, self.colNum))

        self.sub_raster = [[(0, 0), (self.cellWidth/2, 0), (self.cellWidth/2, self.cellHeight/2), (0, self.cellHeight/2)], 
            [(0, self.cellHeight/2), (self.cellWidth/2, self.cellHeight/2), (self.cellWidth/2, self.cellHeight), (0, self.cellHeight)],
            [(self.cellWidth/2, 0), (self.cellWidth, 0), (self.cellWidth, self.cellHeight/2), (self.cellWidth/2, self.cellHeight/2)],
            [(self.cellWidth/2, self.cellHeight/2), (self.cellWidth, self.cellHeight/2),( self.cellWidth, self.cellHeight), (self.cellWidth/2, self.cellHeight)]]
        self.translateSubRaster()
        # create a 2D matrix filled with 0's for the raster
        self.resetRaster()
    
    # center a given polygon
    def centerPolygon(self, poly):
        # polygon translate offsets
        translate_poly = [poly.centroid.x*-1, poly.centroid.y*-1]
        # translate polygon to center at the origin
        return affinity.translate(poly, translate_poly[0], translate_poly[1])
    
    def setMbr(self, global_mbr):
        self.mbr=global_mbr
        self.width=abs(self.mbr[0]-self.mbr[2])
        self.height=abs(self.mbr[1]-self.mbr[3])

    # calculate offset to traslate the grid center to origin point
    def setOffsetFunction(self):
        centroid = [((self.mbr[2]-self.mbr[0])/2)+self.mbr[0], ((self.mbr[3]-self.mbr[1])/2)+self.mbr[1]]
        self.offsetFunction = [(centroid[0])*-1, (centroid[1])*-1]
    
    # translate grid mbr 
    def translateGrid(self):
        self.mbr = [self.mbr[0]+self.offsetFunction[0], 
        self.mbr[1]+self.offsetFunction[1],
        self.mbr[2]+self.offsetFunction[0],
        self.mbr[3]+self.offsetFunction[1]]

    # translate sub cells by centering the grid to (0, 0)
    def translateSubRaster(self):
        for i in range(len(self.sub_raster)):
            for j in range(len(self.sub_raster[i])):
                tmp = list(self.sub_raster[i][j])
                tmp[0] = tmp[0]+self.mbr[0]
                tmp[1] = tmp[1]+self.mbr[1]
                self.sub_raster[i][j] = tuple(tmp)
    
    # reset raster
    def resetRaster(self):
        self.raster = np.zeros((self.rowNum, self.colNum))


    # generate vector by polygon overlay with grid sub cells
    def generateVectorBySubCells(self, poly, foreground=1, icount=1):
        polyT=self.centerPolygon(poly)

        offset_w = 0.0
        offset_h = 0.0
        for i in range(self.rowNum):
            offset_h = 0.0
            for j in range(self.colNum):
                self.raster[i,j]=0
                intersect_count=0                
                for k in range(4):
                    sub_cell=Polygon([(self.sub_raster[k][0][0]+offset_w, self.sub_raster[k][0][1]+offset_h), 
                    (self.sub_raster[k][1][0]+offset_w, self.sub_raster[k][1][1]+offset_h), 
                    (self.sub_raster[k][2][0]+offset_w, self.sub_raster[k][2][1]+offset_h), 
                    (self.sub_raster[k][3][0]+offset_w, self.sub_raster[k][3][1]+offset_h)])
                    # print("{} {}".format(sub_cell, sub_cell.intersects(poly)))
                    if sub_cell.intersects(polyT):
                        intersect_count+=1
                offset_h += self.cellHeight
                if(intersect_count >= icount):
                    self.raster[i,j]+=foreground
            offset_w += self.cellWidth

    # generate vector by the intersecting ratio in each cell
    def generateVectorByCellIntersetingArea(self, poly, foreground=1, ratio=0.75):
        polyT=self.centerPolygon(poly)
        
        cellArea = self.cellHeight*self.cellWidth
        offset_w = 0.0
        offset_h = 0.0
        for i in range(self.rowNum):
            offset_h = 0.0
            for j in range(self.colNum):
                self.raster[i,j]=0
                cell=Polygon([(self.sub_raster[0][0][0]+offset_w, self.sub_raster[0][0][1]+offset_h), 
                    (self.sub_raster[2][1][0]+offset_w, self.sub_raster[2][1][1]+offset_h), 
                    (self.sub_raster[3][2][0]+offset_w, self.sub_raster[3][2][1]+offset_h), 
                    (self.sub_raster[1][3][0]+offset_w, self.sub_raster[1][3][1]+offset_h)])
                if cell.intersects(polyT):
                    intersectArea = cell.intersection(polyT).area
                    if intersectArea/cellArea >= ratio: 
                        self.raster[i,j]+=foreground

                offset_h += self.cellHeight
            offset_w += self.cellWidth

    # generate vector. Mark 1 when polygon touches a cell
    def generateVectorByCellIntersetingArea(self, poly, foreground):      
        offset_w = 0.0
        offset_h = 0.0
        for i in range(self.rowNum):
            offset_h = 0.0
            for j in range(self.colNum):
                self.raster[i,j]=0
                cell=Polygon([(self.sub_raster[0][0][0]+offset_w, self.sub_raster[0][0][1]+offset_h), 
                    (self.sub_raster[2][1][0]+offset_w, self.sub_raster[2][1][1]+offset_h), 
                    (self.sub_raster[3][2][0]+offset_w, self.sub_raster[3][2][1]+offset_h), 
                    (self.sub_raster[1][3][0]+offset_w, self.sub_raster[1][3][1]+offset_h)])
                if cell.intersects(poly):
                    self.raster[i,j]+=foreground
                offset_h += self.cellHeight
            offset_w += self.cellWidth
    
    # visualize grid and polygon overlay
    def plotPolygonGridOverlay(self, poly, extent=[], print_mbr=False, print_min_grid=False):
        px,py = poly.exterior.xy
        plt.plot(px,py)
        c = poly.centroid
        plt.plot(c.x, c.y, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="blue")

        # print MBR of the polygon
        if(print_mbr):
            mbr = poly.bounds
            print("mbr={}".format(mbr))
            mpx = [mbr[0], mbr[2], mbr[2], mbr[0], mbr[0]]
            mpy = [mbr[1], mbr[1], mbr[3], mbr[3], mbr[1]]
            plt.plot(mpx, mpy, color="red")


        if extent:
            # zoom into small region to visualize small polygons
            plt.xlim([extent[0], extent[2]])
            plt.ylim([extent[1], extent[3]])

        # grid
        grid_cord = []
        offset_w = 0.0
        offset_h = 0.0
        for i in range(self.rowNum):
            offset_h = 0.0
            for j in range(self.colNum):
                intersect_count=0
                for k in range(4):
                    sub_cell=Polygon([(self.sub_raster[k][0][0]+offset_w, self.sub_raster[k][0][1]+offset_h), 
                    (self.sub_raster[k][1][0]+offset_w, self.sub_raster[k][1][1]+offset_h), 
                    (self.sub_raster[k][2][0]+offset_w, self.sub_raster[k][2][1]+offset_h), 
                    (self.sub_raster[k][3][0]+offset_w, self.sub_raster[k][3][1]+offset_h)])
                    grid_cord.append(sub_cell)
                offset_h += self.cellHeight
            offset_w += self.cellWidth

        for gi in range(len(grid_cord)):
            gx, gy = grid_cord[gi].exterior.xy
            plt.plot(gx, gy, color="black")
        
        # print starting coordinate of the grid
        if(print_min_grid):
            gx, gy = grid_cord[0].exterior.xy
            print("Grid min=({}, {})".format(gx[0], gy[0]))
            plt.plot(gx[0], gy[0], marker="o", markersize=10, markeredgecolor="green", markerfacecolor="red")
        plt.plot(px,py)

    # convert the uniform grid cells into a list of polyogns
    def convertUniGridToPolys(self):
        wkts=[]
        offset_w = 0.0
        offset_h = 0.0
        for i in range(self.rowNum):
            offset_h = 0.0
            for j in range(self.colNum):
                self.raster[i,j]=0
                cell=Polygon([(self.sub_raster[0][0][0]+offset_w, self.sub_raster[0][0][1]+offset_h), 
                    (self.sub_raster[2][1][0]+offset_w, self.sub_raster[2][1][1]+offset_h), 
                    (self.sub_raster[3][2][0]+offset_w, self.sub_raster[3][2][1]+offset_h), 
                    (self.sub_raster[1][3][0]+offset_w, self.sub_raster[1][3][1]+offset_h)])
                wkts.append(cell)
                offset_h += self.cellHeight
            offset_w += self.cellWidth
            
        return wkts
