import sys
import re
import argparse
import heapq
import subprocess
import zipfile
from io import BytesIO

import streamlit as st

from math import sqrt
import math
import pandas as pd
import numpy as np
np.set_printoptions(precision=6)

from skimage.feature import peak_local_max
from scipy import interpolate

from matplotlib import pyplot as plt
import matplotlib.colors as colors

import rich
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import track
from rich.layout import Layout
from rich.tree import Tree
from rich.status import Status
from rich.rule import Rule
 


class Dat2D():
    def __init__(self, frame, x, y, t, x_distribution, y_distribution) -> None:
        self.frame = frame
        self.x = x
        self.y = y
        self.time = t
        self.x_distribution = x_distribution
        self.y_distribution = y_distribution


class MD2DDat():
    def __init__(self, fp, dt, xnum, ynum) -> None:
        self.frame = None
        self.x = None
        self.y = None
        self.time = None
        self.x_distribution = None
        self.x_bar = None
        self.y_distribution = None
        self.y_bar = None
        self.__ReadDat__(fp)
        self.__TimeSeries__(dt)
        self.__Distribution__(xnum, ynum)
            
    def __TimeSeries__(self, dt):
        t = {"ps":[],"ns":[]}
        for i in self.frame:
            t["ps"].append(i*dt/1000)
            t["ns"].append(i*dt)
        self.time = np.array(t["ns"])
    
    
    def __ReadDat__(self, fp):
        dat = pd.read_csv(fp,sep = ",",header= None)
        self.frame = dat[0].to_numpy()
        self.x = dat[1].to_numpy()
        self.y = dat[2].to_numpy()

    def __Distribution__(self, xnum, ynum):

        xhist, xbins = np.histogram(self.x, bins = xnum, density= False)
        xout = []
        for i, i_t in enumerate(xbins[:-1]):
            x_ = (xbins[i+1] + i_t)/2
            xout.append(x_)
        self.x_distribution = (np.array(xout), xhist)
        self.x_bar =  (max(self.x) - min(self.x)) / xnum
        yhist, ybins = np.histogram(self.y, bins = ynum, density= False)
        yout = []
        for i, i_t in enumerate(ybins[:-1]):
            y_ = (ybins[i+1] + i_t)/2
            yout.append(y_)
        self.y_distribution = (np.array(yout), yhist)
        self.y_bar =  (max(self.y) - min(self.y)) / ynum


    # 此方法用于暴露类中的数据
    def values(self):
        var = Dat2D(self.frame, self.x, self.y, self.time, self.x_distribution, self.y_distribution)
        return var


class Mesh2D():
    def __init__(self, x_center,y_center,s,dx,dy, frame, Pmt) -> None:
        self.x = x_center
        self.y = y_center
        self.s = s
        self.dx = dx
        self.dy = dy
        self.frame = frame
        self.Pmt = Pmt


class GenPMatrix2D():

    def __init__(self, MD2DDat, xnum, ynum) -> None:
        self.mddat2d = MD2DDat.values()
        self.mesh2d = Mesh2D(*self.__GenMesh2D__(xnum, ynum), None, None)

    def __GenMesh2D__(self, xnum, ynum):
        x_min,x_max = self.mddat2d.x.min(), self.mddat2d.x.max()
        y_min,y_max = self.mddat2d.y.min(), self.mddat2d.y.max()
        dx = (x_max-x_min) / (2*xnum)
        dy = (y_max-y_min) / (2*ynum)
        x_center = np.array([])
        y_center = np.array([])
        func = lambda x,d,c0 : (2*x-1)*d + c0
        s = 2*dx*2*dy           
        for i in range(1,xnum+1):
            x_center = np.append(x_center,func(i,dx,x_min))
        for i in range(1,ynum+1):
            y_center = np.append(y_center,func(i,dy,y_min))
        return (x_center,y_center,s,dx,dy)
    
    def __fallW__(self, mesh_xi, mesh_yi):
        
        mesh_x = self.mesh2d.x[mesh_xi]
        mesh_y = self.mesh2d.y[mesh_yi]
        dx = self.mesh2d.dx
        dy = self.mesh2d.dy
        
        # 考虑通过bool索引
        # 构造DF
        XY = pd.DataFrame({"x":self.mddat2d.x, "y":self.mddat2d.y})
        XY_1 = XY[((mesh_x - dx) <= XY["x"]) & (XY["x"] < (mesh_x + dx)) & ((mesh_y - dy) <= XY["y"]) & (XY["y"] < (mesh_y + dy))]
        # 获取符合要求的与不符合要求的
        XY_1_index = set(XY_1.index.to_list())
        XY_0_index = set(XY.index.to_list()) - XY_1_index
        # 计数与统计构象
        count = len(XY_1_index)
        XY_1_index = list(XY_1_index)
        XY_0_index = list(XY_0_index)
        frame = self.mddat2d.frame[XY_1_index]
        # 更新数据集
        self.mddat2d.x = self.mddat2d.x[XY_0_index]
        self.mddat2d.y = self.mddat2d.y[XY_0_index]
        self.mddat2d.frame = self.mddat2d.frame[XY_0_index]
        self.mddat2d.time =  self.mddat2d.time[XY_0_index]
        return (count,frame)
        
    
    def Pmatrix(self):
        # 总采样点
        N = len(self.mddat2d.x)
        if not self.mesh2d.frame:
            # 概率密度矩阵
            self.mesh2d.Pmt = np.zeros(shape=[len(self.mesh2d.x),len(self.mesh2d.y)])

        # 创建dict储存采样网格原始frame信息，其中keys的格式为"i-j"
        if not self.mesh2d.frame:
            self.mesh2d.frame = {}

        rprint("Probability Density Calculation")
        for i in track(range(len(self.mesh2d.x)),description = "Runing..."):
            for j in range(len(self.mesh2d.y)):
                fallWout = self.__fallW__(i,j)
                # 计算该点的概率密度
                P = fallWout[0] / (N*self.mesh2d.s)
                # 记录概率密度
                self.mesh2d.Pmt[i][j] = P
                # 记录每个点的采样构象
                self.mesh2d.frame["{}-{}".format(i,j)] = fallWout[1]
       
        old_max = self.mesh2d.Pmt.max()
        old_min = self.mesh2d.Pmt.min()
        tol_density = np.sum(self.mesh2d.Pmt)
        self.mesh2d.Pmt /=tol_density
        new_max = self.mesh2d.Pmt.max()
        new_min = self.mesh2d.Pmt.min()

        #table = Table()
        #table.add_column("[yellow]Area of window")
        #table.add_column("[blue]Total Numbers of point")
        #table.add_column("")
        #table.add_column("[yellow]Original")
        #table.add_column("[blue]Normalization")
        #table.add_row("{:.2f}".format(self.mesh2d.s), "{}".format(N), "Maximum Value", "{:.4f}".format(old_max), "{:.4f}".format(new_max))
        #table.add_row(""                            , ""            , "Minimum value", "{:.4f}".format(old_min), "{:.4f}".format(new_min))
        #rprint(table)

class PPeak():
    def __init__(self,n_peak,mesh2d) -> None:
        self.x = []
        self.y = []
        self.mesh2d = mesh2d
        self.ij = peak_local_max(self.mesh2d.Pmt,num_peaks=n_peak)
        self.G = []
        self.__PeakXY__()
    def __PeakXY__(self):
        for i in range(self.ij.shape[0]):
            self.x.append(self.mesh2d.x[self.ij[i,0]])
            self.y.append(self.mesh2d.y[self.ij[i,1]])
    # 用于删除指定索引
    def delPeak(self,i_list):
        x, y, G = [], [], []
        ij = np.empty(shape=[0, 2],dtype="int")
        for i,x_ in enumerate(self.x):
            if i not in i_list:
                x.append(x_)
                y.append(self.y[i])
                G.append(self.G[i])
                ij = np.append(ij,values=[self.ij[i]], axis=0)
        self.x = x
        self.y = y
        self.ij = ij
        self.G = G
    def setG(self,G):
        self.G = G


class PeakPATH():
    def __init__(self, Gbbis) -> None:
        self.graph1, self.graph2 = None, None
        self.__cSign__(Gbbis)
    
    # 构造权重字典，两次
    ## graph1: Ea -> Eb = Eb
    ## graph2: Ea -> Eb = Eb - Ea
    def __cSign__(self,Z):
        dic1 = {}    
        dic2 = {} 
        ij = Z.shape
        for i in range(ij[0]):
            for j in range(ij[1]):
                dic1["{}_{}".format(i,j)] = {}
                dic2["{}_{}".format(i,j)] = {}
                if i+1 in range(ij[0]) and j-1 in  range(ij[1]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i+1,j-1)] =  Z[i+1][j-1]
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i+1,j-1)] =  Z[i+1][j-1] - Z[i][j]
                if i+1 in range(ij[0]) and j+1 in  range(ij[1]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i+1,j+1)] =  Z[i+1][j+1] 
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i+1,j+1)] =  Z[i+1][j+1] - Z[i][j]
                if i+1 in range(ij[0]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i+1,j )]=  Z[i+1][j]
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i+1,j )]=  Z[i+1][j]  - Z[i][j]
                if i-1 in range(ij[0]) and j-1 in range(ij[1]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i-1,j-1)] = Z[i-1][j-1] 
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i-1,j-1)]=  Z[i-1][j-1]  - Z[i][j]
                if i-1 in range(ij[0]) and j+1 in range(ij[1]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i-1,j+1)] = Z[i-1][j+1]
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i-1,j+1)]=  Z[i-1][j+1] - Z[i][j]
                if i-1 in range(ij[0]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i-1,j)] = Z[i-1][j]
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i-1,j)]=  Z[i-1][j] - Z[i][j]
                if j-1 in range(ij[1]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i,j-1)] =   Z[i][j-1] 
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i,j-1)]=   Z[i][j-1]  - Z[i][j]
                if j+1 in range(ij[1]):
                    dic1["{}_{}".format(i,j)]["{}_{}".format(i,j+1)] =   Z[i][j+1]
                    dic2["{}_{}".format(i,j)]["{}_{}".format(i,j+1)]=  Z[i][j+1] - Z[i][j]
        self.graph1 = dic1
        self.graph2 = dic2


    def __init_distance__(self, graph, start):
        distance = {start :0}
        for vertex in graph:
            if vertex != start:
                distance[vertex] = math.inf
        return distance
    
    def dijkstra(self, start):
        pqueue = []
        heapq.heappush(pqueue, (0, start))
        seen = set()
        parent = {start: None}
        distance = self.__init_distance__(self.graph1, start)
    
        while (len(pqueue) > 0):
            pair = heapq.heappop(pqueue)
            dist = pair[0]
            vertex = pair[1]
            seen.add(vertex)
            nodes = self.graph1[vertex].keys()
            for w in nodes:
                if w not in seen:
                    if dist + self.graph1[vertex][w] < distance[w]:
                        heapq.heappush(pqueue, (dist + self.graph1[vertex][w], w))
                        parent[w] = vertex
                        distance[w] = dist - self.graph1[vertex][w]
        return parent, distance

    def Traj(self, start, end, parent):
        traj = []
        #倒着输出的
        old = end
        while end != None:
            traj.append([eval(i) for i in end.split("_")])
            end = parent[end]
        return traj

    def TrajDis(self,traj):     
        out = 0
        for i in range(1,len(traj)):
            start = traj[i-1]
            end = traj[i]
            dis = self.graph2["{}_{}".format(*start)]["{}_{}".format(*end)]
            out += dis
        return out


class PeakRepDat():
    def __init__(self,min_, max_, ave_, med_,  distribution_x,  distribution_y):
        self.min_ = min_
        self.max_ = max_
        self.ave_ = ave_
        self.med_ = med_
        self.distribution_x = distribution_x
        self.distribution_y = distribution_y

        self.DatTime = None
    def Frame2Time(self, dt):
        self.DatTime = [
             self.min_ * dt
            ,self.max_ * dt
            ,self.ave_ * dt
            ,self.med_ * dt
            ,self.distribution_x * dt
            ,self.distribution_y
        ]
    def Pinfo(self,peak, x, y):
        out_ss = ""
        distribution_max_i = np.argmax(self.distribution_y)
        out_ss += "{}: ({:.2f},{:.2f})\n".format(peak,x,y)

        out_ss += "Frame\n"
        out_ss += "\tMaximum value: {}\n".format(int(self.max_))
        out_ss += "\tMinimum value: {}\n".format(int(self.min_))
        out_ss += "\tAverage value: {:.2f}\n".format(self.ave_)
        out_ss += "\tMedian value: {:.2f}\n".format(self.med_)
        out_ss += "\tMaximum probability distribution: {:.2f}\n".format(self.distribution_x[distribution_max_i])
        out_ss += "\tDistribution count: {}\n".format(int(self.distribution_y[distribution_max_i]))
        if self.DatTime != None:
            out_ss += "Time (ns)\n"
            out_ss += "\tMaximum value: {:.2f} ns\n".format(int(self.DatTime[1]))
            out_ss += "\tMinimum value: {:.2f} ns\n".format(int(self.DatTime[0]))
            out_ss += "\tAverage value: {:.2f} ns\n".format(self.DatTime[2])
            out_ss += "\tMedian value: {:.2f} ns\n".format(self.DatTime[3])
            out_ss += "\tMaximum probability distribution: {:.2f} ns\n".format(self.DatTime[4][distribution_max_i])
            out_ss += "\tDistribution count: {} ns\n".format(int(self.DatTime[5][distribution_max_i]))
        return out_ss


class PeakCluster():
    def __init__(self, mddat2d) -> None:
        self.mddat2d = mddat2d
        self.KNNdata = None

    # 20240118 更新：首先对输入的xy进行归一化算法,应该在各自维度
    def KNN(self, peak, cutoff):
        label = []
        cutbool = []
    
        sample_x_max = self.mddat2d.x.max()
        sample_x_min = self.mddat2d.x.min()
        sample_y_max = self.mddat2d.y.max()
        sample_y_min = self.mddat2d.y.min()        
        xpeak = (peak.x - sample_x_max)/(sample_x_max - sample_x_min)
        ypeak = (peak.y - sample_y_max)/(sample_y_max - sample_y_min)
        # 这俩应该是numpy 
        x_sample = (self.mddat2d.x - sample_x_max)/(sample_x_max - sample_x_min)
        y_sample = (self.mddat2d.y - sample_y_max)/(sample_y_max - sample_y_min)

        f_sample = self.mddat2d.frame

        Fdist = lambda x0,y0,x1,y1: math.sqrt(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2))

        for i_, x_ in enumerate(x_sample):
            d = {}
            for ip_,xp_ in enumerate(xpeak):
                d[str(ip_)] = Fdist(xp_, ypeak[ip_], x_, y_sample[i_])

            d_ = sorted(d.items(),  key=lambda d: d[1], reverse=False)[0]
            label_s = eval(d_[0])
            label.append(label_s)
            dist_s  = d_[1]
            if dist_s <= cutoff:
                cutbool.append(1)
            else:
                cutbool.append(0)

        self.KNNdata = self.__labelMap__(label , cutbool)

        return label, cutbool
   
    def __labelMap__(self, label , cutbool):
        return pd.DataFrame({ "frame":self.mddat2d.frame
                             , "x":self.mddat2d.x
                             , "y":self.mddat2d.y
                             , "label":label
                             , "cutbool":cutbool})

    def __distribution1D__(self, data, nbins):
        out = []
        hist, bins = np.histogram(data, bins = nbins, density= False)
        for i, i_t in enumerate(bins[:-1]):
            var = (bins[i+1] + i_t)/2
            out.append(var)
        
        return np.array(out), hist

    # 被KNNrep调用，用于计算每一类构象群的代表构象
    def __rep__(self, Frames, nbins):
        mean_ = sum(Frames) / len(Frames)
        std = np.std(np.array(Frames))

        if (len(Frames) - 1)%2 != 0:
            var = len(Frames) - 2          # 偶数个
        else:
            var = len(Frames) - 1          # 奇数个
    
        rep = Frames[int(var/2)]
        ave = sum(Frames)/len(Frames)
        min_ = min(Frames)
        max_ = max(Frames)

        distribution = self.__distribution1D__(Frames, nbins)
    
        return PeakRepDat(min_, max_, ave, rep, distribution[0], distribution[1])

    def KNNrep(self, nbins):
        dic = {}
        # 读取KNNdata
        PeakSet = sorted(set(self.KNNdata["label"]))
        out_ss = ""
        for p in PeakSet:
            p_frame = self.KNNdata[self.KNNdata["label"] == p]["frame"].to_list()
            out_ss += "P{}:\n".format(p)
            var  = [str(j) for j in p_frame]
            out_ss += (",".join(var)+"\n")
            out_ss += "END\n"
            dic[str(p)] = self.__rep__(p_frame, nbins)
        return dic, out_ss

    # 输出每个构象微状态的代表构象信息
    def Prep(self,dic,peak,dt):
        out_ss = ""
        rep_keys = sorted(dic.keys())
        for i in rep_keys:
            var = dic[i]
            var.Frame2Time(dt)
            out_ss += var.Pinfo("P{}".format(i), peak.x[eval(i)],peak.y[eval(i)])
        return out_ss
    def KNNtransfer(self, T):

        dat = self.KNNdata.copy(deep=True)
        # 构建转移矩阵
        #    - Cmatrix: 计数矩阵
        #    - Tmatrix: 转移概率矩阵
        n_state = len(set(dat["label"].to_list()))
        Cmatrix = np.zeros(shape=(n_state, n_state))
        n_tran = 0
        #rprint("Calculation of Transition Matrix between States")
        for t in track(range(1,T+1),description = "Runing..."):
            # 获得子序列
            sub_df = dat.iloc[::t,:]
            # 重排索引
            sub_df = sub_df.reset_index(drop=True)
            for i in range(sub_df.shape[0]-1):
                stata_1 = sub_df.iloc[i,3]
                stata_2 = sub_df.iloc[i+1,3]
                Cmatrix[stata_1][stata_2] += 1
                n_tran += 1
        return pd.DataFrame(Cmatrix/n_tran)
    
    def Ptransfer(self, TranferMatrax):
        out_ss = ""
        shape_x, shape_y = TranferMatrax.shape
        for i in range(shape_x):
            for j in range(shape_y):
                P_i2j = TranferMatrax.iloc[i,j]
                out_ss += "P(state[{}]->state[{}])={:.2f}%\t".format(i,j,P_i2j*100)
                if j == (shape_y-1):
                    out_ss += "\n"
        return out_ss

def CalGibbs(P_martix, Peak, T):                        # 20230616 T参数新增
    k = 1.3806505*10**(-23)
    C = 0 
    c1 = 4186
    NA = 6.023*10**(23)                                 # 20230426: SB ......... kT ->kcal/mol
    # flatten
    shape = P_martix.shape
    P_ = P_martix.flatten()
    
    # 概率密度为0的密度点的Gibbs会直接变为0,P < 1，所以
    Gibbs = lambda P: np.nan if abs(P - 0) <= 1e-10  else ((-k*T*np.log(P) + C)/c1)*NA
    G = np.array(list(map(Gibbs,P_))).reshape(shape)
    var = G[~np.isnan(G)]
    G_min = var.min()
    G_max = var.max()
    # 这一步是为了让概率密度最大处的G为0，计算完毕后，需要再计算一次max,min
    G = G-G_min
    var = G[~np.isnan(G)]
    G_min = var.min()
    G_max = var.max()
    G = np.nan_to_num(G, nan=G_max + (G_max-G_min)/10)
    #G = np.nan_to_num(G, nan=500)

    # 提取Peak中的峰值点的G
    ij = Peak.ij
    g_peak = []
    for i in range(ij.shape[0]):
        g_peak.append(G[ij[i,0],ij[i,1]])
    
    out_ss = ""
    out_ss += "Free Energy\n"
    out_ss +="Original Max Value: {:.4f}".format(G_max)
    out_ss +="Original Min Value: {:.4f}".format(G_min)
    out_ss +="Transformed Value: {:.4f}".format(G_max + (G_max-G_min)/10)
    out_ss +="Transformed Max Value: {:.4f}".format(G.max())
    out_ss +="Transformed Min Value: {:.4f}".format(G.min())
    
    return G,g_peak, out_ss

# 插值算法
class GdDInter():

    def __init__(self, mesh2d, Gbbis) -> None:
    
        self.mesh2d = mesh2d
        self.Gbbis = Gbbis
        
    def __IntNum__(self, xnum, ynum):
        xmin = self.mesh2d.x.min()
        xmax = self.mesh2d.x.max()
        ymin = self.mesh2d.y.min()
        ymax = self.mesh2d.y.max()

        # 原始的网格矩阵
        mesh2dX,mesh2dY = np.meshgrid(self.mesh2d.x,self.mesh2d.y)
        # 新生成的网格矩阵
        xx,yy = np.meshgrid(np.linspace(xmin,xmax,num = xnum,endpoint=True)
                            ,np.linspace(ymin,ymax,num = ynum,endpoint=True))
        # 插值获得新生成网格矩阵的Z值
        zz = interpolate.griddata((mesh2dX.flatten(),mesh2dY.flatten())
                                  ,self.Gbbis.flatten()
                                  ,(xx,yy)
                                  , method='linear',fill_value =0)
        return xx, yy, zz                                           # 返回的都是矩阵

    def run(self, xnum, ynum):
        out_ss = ""
        #rprint("[bold yellow]4. Griddate interpolation Module")
        #rprint("The data will be saved as \"Gbbis-Griddate.csv\"")
        old_max = self.Gbbis.flatten().max()
        old_min = self.Gbbis.flatten().min()
        #rprint("Gibbs_MAX={:.2f},Gibbs_MIN={:.2f}".format(old_max, old_min))
        X,Y,Z = self.__IntNum__(xnum, ynum)
        new_max = self.Gbbis.flatten().max()
        new_min = self.Gbbis.flatten().min()
  
        out_ss += "Free Energy:\n"
        out_ss += "Original Maximum value: {:.2f}".format(old_max)
        out_ss += "Original Minimum value: {:.2f}".format(old_min)
        out_ss += "Interpolation by Griddata Maximum value: {:.2f}".format(new_max)
        out_ss += "Interpolation by Griddata Minimum value: {:.2f}".format(new_min)
    
        #pd.DataFrame([X.flatten(),Y.flatten(),Z.flatten()]).T.to_csv("Gbbis-Griddate.csv")
        return X,Y,Z,out_ss
    
    # 此函数用于从插值后矩阵中找到要求的势能面极小值，只用于画图数据，因此输出的是x y而不是索引，返回的是一个字典，储存着该点的x y
    def FindPeak(self,Xpeak,Ypeak, XintNum, YintNum):
        dic = {}
        for i_px, px in enumerate(Xpeak):
            py = Ypeak[i_px]
            d_x2 = (np.array(XintNum) - px)**2
            d_y2 = (np.array(YintNum) - py)**2
            dis = np.sqrt(d_x2 + d_y2)
            i_min = np.argmin(dis)
            dic[str(i_px)] = (XintNum[i_min], YintNum[i_min])
        return dic



# 以下是绘图部分
## 全局字体设置
from matplotlib.font_manager import FontProperties
font_path="/home/hang/MYscrip/font/times.ttf"
font_prop =  FontProperties(fname=font_path)
plt.switch_backend("agg")
plt.rcParams["axes.labelweight"] ="bold"
plt.rcParams["font.family"]=font_prop.get_name()
plt.rcParams["font.weight"]="bold"
plt.rcParams["font.size"]=10

def drawDatDistribution(x, y, label,x_bar):
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    
    ax.tick_params(
         which='both'
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6 
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
    ax.set_xlabel("{}".format(label))
    ax.set_ylabel("Count")
    ax.set_xlim(x.min(),x.max())
    ax.set_ylim(y.min(),y.max()+(y.max()-y.min())/10)
    
    plt.plot(x,y,alpha = 1,linewidth = 2, color = "#D57B70")
    # 加入fill
    plt.bar(x,y,x_bar, color="#1399B2", linewidth=0.5,edgecolor="green")
    #plt.fill_between(x, 0 ,y, facecolor = "darkgreen")
    plt.tight_layout()
    st.pyplot(fig)
    return fig
   
def draw2DScatterTime(md2ddat,xlabel,ylabel):
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.tick_params(
         which='both'
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6 
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.set_xlabel("{}".format(xlabel))
    ax.set_ylabel("{}".format(ylabel))
    ax.set_xlim(md2ddat.x.min(),md2ddat.x.max())
    ax.set_ylim(md2ddat.y.min(),md2ddat.y.max())
    plt.scatter(md2ddat.x,md2ddat.y,alpha = 0.7,s= 5,c=md2ddat.time)
    cbar = plt.colorbar()
    cbar.set_label('Time (ns)')
    plt.tight_layout()
    st.pyplot(fig)
    return fig

def draw2DScatterCircle(md2ddat,slabel,myPeak, cutoff,xlabel,ylabel):
    
    dat = pd.DataFrame({"x":md2ddat.x,"y":md2ddat.y,"label":slabel})
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.tick_params(
         which='both'
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6 
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.set_xlabel("{}".format(xlabel))
    ax.set_ylabel("{}".format(ylabel))
    ax.set_xlim(md2ddat.x.min(),md2ddat.x.max())
    ax.set_ylim(md2ddat.y.min(),md2ddat.y.max())
    
    for i in set(slabel):
        Datai = dat[dat["label"] == i]
        plt.scatter(Datai["x"],Datai["y"],alpha = 0.7,s= 5)
    # 绘制峰值点:
    for i,x_ in enumerate(myPeak.x):
        plt.scatter(x_,myPeak.y[i],color = "black")
        plt.text(x_,myPeak.y[i],s="P{}".format(i))
        # 并绘制cutoff
        draw_circle = plt.Circle((x_,myPeak.y[i]),cutoff,fill=False)
        plt.gcf().gca().add_artist(draw_circle)
    plt.tight_layout()
    st.pyplot(fig)
    return fig

def draw2DFEL(X, Y, Z, cname ,select_plot, xlabel, ylabel):
    # 绘制概率密度曲线
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.tick_params(
         which='both'
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6 
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)



    ax.set_xlim(X.min(),X.max())
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlabel("{}".format(xlabel))
    ax.set_ylabel("{}".format(ylabel))

    C=plt.contour(X,Y,Z,5,colors='black',linewidths= 0.2)  #生成等值线图
    plt.contourf(X,Y,Z,5,alpha=0.2)
    Pcolor= plt.pcolor(X,Y,Z
                       ,shading='auto',cmap=cname
                       ,norm = colors.TwoSlopeNorm(vmin=Z.min(), vcenter=(Z.max()-Z.min())/2, vmax=Z.max()))
    plt.colorbar(Pcolor)

    plt.scatter(select_plot[0], select_plot[1], color = "black",s = 6)
    plt.tight_layout()
    st.pyplot(fig)
    return fig

def drawPeakTimeDistribution(dic,dt, xlabel, ylabel):
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.tick_params(
         which='both'
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6 
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)


    #ax.spines['bottom'].set_linewidth(4)
    #ax.spines['top'].set_linewidth(4)
    #ax.spines['left'].set_linewidth(4)
    #ax.spines['right'].set_linewidth(4)
    #ax.set_xlim(X.min(),X.max())
    #ax.set_ylim(Y.min(),Y.max())
    ax.set_xlabel("{}".format(xlabel))
    ax.set_ylabel("{}".format(ylabel))
    
    rep_keys = sorted(dic.keys())
    for i in rep_keys:
        peak = dic[i]
        peak.Frame2Time(dt)
        i_peak_x = peak.DatTime[4]
        i_peak_y = peak.distribution_y
        plt.plot(i_peak_x, i_peak_y,label = "P{}".format(i))
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)
    return fig

def drawPATH(X,Y,Z,cname,traj_x, traj_y,select_plot,xlabel, ylabel):
    # 绘制概率密度曲线
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.tick_params(
         which='both'
        ,left=True
        ,direction='out'
        ,width=2 
        ,length=6 
    )
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    #ax.spines['bottom'].set_linewidth(4)
    #ax.spines['top'].set_linewidth(4)
    #ax.spines['left'].set_linewidth(4)
    #ax.spines['right'].set_linewidth(4)

    ax.set_xlim(X.min(),X.max())
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlabel("{}".format(xlabel))
    ax.set_ylabel("{}".format(ylabel))

    C=plt.contour(X,Y,Z,5,colors='black',linewidths= 0.2)  #生成等值线图
    plt.contourf(X,Y,Z,5,alpha=0.2)
    Pcolor= plt.pcolor(X,Y,Z
                       ,shading='auto',cmap=cname
                       ,norm = colors.TwoSlopeNorm(vmin=Z.min(), vcenter=(Z.max()-Z.min())/2, vmax=Z.max()))
    plt.colorbar(Pcolor)
    
    # 绘制PATH
    plt.plot(traj_x,traj_y,color = "white", linestyle = "--", linewidth = 1.5)
    # 起点与终点
    if select_plot != None:
        select_x = select_plot[0]
        select_y = select_plot[1]

    plt.scatter(select_x, select_y, c = "black",s = 6)
    plt.tight_layout()
    st.pyplot(fig)
    return fig


## 绘制空图用于初始化界面
def drawNULL():
    fig,ax = plt.subplots(dpi=300)
    fig.patch.set_facecolor('gray')
    ax.patch.set_facecolor('gray')
    plt.axis("off")
    st.pyplot(fig)

def convertimg(fig):
    buf = BytesIO()
    fig.savefig(buf,format="PNG")
    buf.seek(0)
    return buf.getvalue()


def downDATA(column,object_dict):

    def __File__(object_dict):
        buf = BytesIO()
        with zipfile.ZipFile(buf, "x") as all_zip:
            keys = object_dict.keys()
            for fn in keys:

                all_zip.writestr(fn, object_dict[fn])
        return buf
    with column[0]:       
        st.download_button(
            label='Download',
            key='d1',
            data=__File__(object_dict),
            file_name='all.zip',
            mime='application/zip',
            use_container_width = True
            )

def printweb(ssname,txt):
    with st.session_state[ssname]:
        st.code(txt, language='python')

def main():
    st.set_page_config(
        page_title = "FE2D by IAW [HENU]"
        ,layout = "wide"
    )

    st.title("FE2D")

    # 采用表单，固定参数写入到侧边栏
    with st.sidebar.form('my_form'):
        col1, col2 = st.columns(2)
        with col1:
            dimX = st.number_input("**DimX**", format='%d',step = 5)
            dimX = int(dimX)
        with col2:
            dimY = st.number_input("**DimY**", format='%d',step = 5)
            dimY = int(dimY)

        col3, col4 = st.columns(2)
        with col3:
            smoX = st.number_input("**SmoX**", format='%d',step = 5)
            smoX = int(smoX)
        with col4:
            smoY = st.number_input("**SmoY**", format='%d',step = 5)
            smoY = int(smoY)
        dt = st.number_input("**Time conversion factor (ns/line)**", step = 0.001)
        SimulatedTemperature = st.number_input("**Simulated Temperature (K)**", step = 20)
        col5, col6 = st.columns(2)
        with col5:
            npeak = st.number_input("**Number of Peaks**", format='%d',step = 1)
            npeak = int(npeak)
        with col6:
            cutoff = st.number_input("**Cutoff**",step = 0.5)
        cmp = st.text_input("**Matplotlib Color:**") 
        col7, col8 = st.columns(2)
        with col7:
            xlabel = st.text_input("**X-axis label**")
           
        with col8:
            ylabel = st.text_input("**Y-axis label**")
        
        nbinT = st.number_input("**Number of Transfer Sub-traj bins:**", format='%d',step = 10) 
        fp = st.file_uploader("**Data**", type="csv")
        submitted = st.form_submit_button('**Submit**')

    # 设置主界面的显示内容

    col9, col10 = st.columns(2)
    col11, col12 = st.columns(2)
    ss_del_peak = st.text_input(":green[Peak index to be deleted: ]")
    col13, col14 = st.columns(2)
    # 需要输出转移矩阵
    ss_path = st.text_input(":green[Enter the points you want to transfer(e.g. 0-1: P0 -> P1): ]")
    col15, col16 = st.columns(2)

    st.write(":blue[**Program Run Log**]")
    st.session_state['stdOUT'] = st.empty()
    #global image1,image2,image3,image4,stdOUT,txt,PCAT, KNMPEAK,GBBISF
    st.session_state['txt'] = ""
    downtab = st.tabs(["DownLoad File"])
    # 绘制输入数据的概率分布
    with col9:
        st.write(":blue[**X Distribution**]")
        st.session_state['image1'] = st.empty()
        with st.session_state['image1']:
            drawNULL()      
    with col10:
        st.write(":blue[**Y Distribution**]")
        st.session_state['image2'] = st.empty()
        with st.session_state['image2']:
            drawNULL()      
    # 绘制输入数据与时间的变化，以及KNN聚类结果
    with col11:
        st.write(":blue[**Conformational Time Distribution**]")
        st.session_state['image3'] = st.empty()
        with st.session_state['image3']:
            drawNULL()
    with col12:
        st.write(":blue[**KNN Cluster**]")
        st.session_state['image4'] = st.empty()
        with st.session_state['image4']:
            drawNULL()
    # 绘制修改后的峰值图以及原始的FE2D景观图
    with col13:
        st.write(":blue[**KNN Cluster**]")
        st.session_state['image5'] = st.empty()
        with st.session_state['image5']:
            drawNULL()   
    with col14:
        st.write(":blue[**FE2D-1**]")
        st.session_state['image6'] = st.empty()
        with st.session_state['image6']:
            drawNULL()      
    # 绘制插值后的FE2D景观图以及PATH图
    with col15:
        st.write(":blue[**FE2D-1**]")
        st.session_state['image7'] = st.empty()
        with st.session_state['image7']:
            drawNULL()   
    with col16:
        st.write(":blue[**FE2D-PATH**]")
        st.session_state['image8'] = st.empty()
        with st.session_state['image8']:
            drawNULL()      
  
    with st.session_state['stdOUT']:
        st.code(" ", language='python')

    st.session_state["txt"] += "FE2D is initialized successfully\n"
    printweb("stdOUT", st.session_state['txt'])
       

    if fp:
        # 读取数据
        mymd = MD2DDat(fp,dt,dimX, dimY)
        with col9:
            with st.session_state['image1']:
                XdirtributionGraph = drawDatDistribution(*mymd.x_distribution,xlabel, mymd.x_bar)
        with col10:
            with st.session_state['image2']:
                YdirtributionGraph = drawDatDistribution(*mymd.y_distribution,ylabel, mymd.y_bar)

        # 计算概率矩阵
        myP = GenPMatrix2D(mymd, dimX, dimY)
        myP.Pmatrix()
        myPeak = PPeak(npeak,myP.mesh2d)
        # 计算Free Energy
        myG = CalGibbs(myP.mesh2d.Pmt, myPeak, SimulatedTemperature)
        myPeak.setG(myG[1])
        
        # 开始尝试进行KNN聚类
        myPeakKNNTest = PeakCluster(mymd)
        # KNN
        myKnnTestLabel,myKnnTestCut = myPeakKNNTest.KNN(myPeak, cutoff)
        # 输出采样代表构象信息， KNNRepInfoTest用于文件储存
        myKNNPepTest,KNNRepInfoFileTest = myPeakKNNTest.KNNrep(nbinT)
        # 输出信息到监视器
        KNNRepInfoMonitorTest = myPeakKNNTest.Prep(myKNNPepTest, myPeak, dt)
        
        st.session_state["txt"] += KNNRepInfoMonitorTest
        printweb("stdOUT", st.session_state['txt'])
        
        with col11:
            with st.session_state['image3']:
                ScatterTimeGraph = draw2DScatterTime(mymd,xlabel,ylabel)

        with col12:
            with st.session_state['image4']:
                PeakKNNGraph1 = draw2DScatterCircle(mymd
                                                    , myPeakKNNTest.KNNdata["label"].to_list()
                                                    , myPeak, cutoff
                                                    , xlabel, ylabel)
        
        # 获取删除值
        if ss_del_peak != "":
            if ss_del_peak != "-1":
                peak_del_index = [eval(i) for i in ss_del_peak.rstrip("\n").split(",")]
                # 删除去除的peak
                myPeak.delPeak(peak_del_index)
                myPeakKNN = PeakCluster(mymd)
                myKnnLabel,myKnnCut = myPeakKNN.KNN(myPeak, cutoff)
                # 输出采样代表构象信息， KNNRepInfoTest用于文件储存
                myKNNPep,KNNRepInfoFile = myPeakKNN.KNNrep(nbinT)
                # 输出信息到监视器
                KNNRepInfoMonitor = myPeakKNN.Prep(myKNNPep, myPeak, dt)      
            else:
                myPeakKNN = myPeakKNNTest
                myKnnLabel, myKnnCut = myKnnTestLabel, myKnnTestCut
                myKNNPep = myKNNPepTest
           
            # 绘制选择Peak后的图像   
            with col13:
                with st.session_state['image5']:
                    PeakKNNGraph2 = draw2DScatterCircle(mymd
                                                , myPeakKNN.KNNdata["label"].to_list()
                                                , myPeak, cutoff
                                                , xlabel, ylabel)

            # 绘制原始的FE2D-1
            with col14:
                with st.session_state['image6']:
                    FEL2D1 = draw2DFEL(myP.mesh2d.x
                                        , myP.mesh2d.y
                                        , myG[0].T
                                        , cmp
                                        , (myPeak.x, myPeak.y)
                                        , xlabel,ylabel)
            # 计算转移矩阵
            myTmatrax = myPeakKNN.KNNtransfer(T = int(len(mymd.x)/nbinT))
            # 输出转移矩阵
            PtransferOut = myPeakKNN.Ptransfer(myTmatrax)
            st.session_state["txt"] += PtransferOut
            printweb("stdOUT", st.session_state['txt'])

            if ss_path != "":
                start_i, end_i = ss_path.rstrip("\n").split("-")
                start = "{}_{}".format(myPeak.ij[eval(start_i),0], myPeak.ij[eval(start_i),1])
                end =  "{}_{}".format(myPeak.ij[eval(end_i),0], myPeak.ij[eval(end_i),1])

                mypath = PeakPATH(myG[0])

                par,dis = mypath.dijkstra(start)
                traj = mypath.Traj(start, end, par)
                trajdis = mypath.TrajDis(traj)

                # 进行插值算法       
                myFE = GdDInter(myP.mesh2d, myG[0])
                # 插值
                EXintN,FEYintN, FEZintN, GdDInterOut = myFE.run(smoX, smoX)
                # 寻找峰值点
                FEpeakFind = myFE.FindPeak(myPeak.x, myPeak.y, EXintN.flatten(), FEYintN.flatten())
                select_plot_x, select_plot_y = [], []
                for i in sorted([eval(i) for i in FEpeakFind.keys()]):
                    select_plot_x.append(FEpeakFind[str(i)][0])
                    select_plot_y.append(FEpeakFind[str(i)][1])
                with col15:  
                    with st.session_state['image7']:
                        FEL2D2 = draw2DFEL(EXintN
                                            ,FEYintN
                                            ,FEZintN.T
                                            , cmp
                                            , (select_plot_x, select_plot_y)
                                            ,xlabel, ylabel)
                # 寻找PATH的点
                PathPoint = myFE.FindPeak(myP.mesh2d.x[np.array(traj)[:,0]]
                                         , myP.mesh2d.y[np.array(traj)[:,1]]
                                         , EXintN.flatten()
                                         , FEYintN.flatten())
                traj_x, traj_y = [], []
                for i in sorted([eval(i) for i in PathPoint.keys()]):
                    traj_x.append(PathPoint[str(i)][0])
                    traj_y.append(PathPoint[str(i)][1])
                with col16:   
                    with st.session_state['image8']:
                        PATH = drawPATH(EXintN, FEYintN, FEZintN.T
                                        ,cmp 
                                        ,traj_x ,traj_y
                                        ,(select_plot_x, select_plot_y)
                                        ,xlabel, ylabel)

                downDATA(downtab, { "XY-distribution.csv":pd.DataFrame({  "x": mymd.x_distribution[0], "xD":mymd.x_distribution[1]
                                                                       , "y": mymd.y_distribution[0], "yD":mymd.y_distribution[1]}).to_csv()
                                    , "FreeEnergy-InterNum.csv":pd.DataFrame([  EXintN.flatten()
                                                                              , FEYintN.flatten()
                                                                              , FEZintN.T.flatten()]).T.to_csv()
                                    , "FreeEnergy.csv":pd.DataFrame([  np.meshgrid(myP.mesh2d.x,myP.mesh2d.y)[0].flatten()
                                                                     , np.meshgrid(myP.mesh2d.x,myP.mesh2d.y)[1].flatten()
                                                                     , myG[0].T.flatten()]).T.to_csv()
                                    , "XDistribution.png":convertimg(XdirtributionGraph)
                                    , "YDistribution.png":convertimg(YdirtributionGraph) 
                                    , "ScatterTimeGraph.png":convertimg(ScatterTimeGraph)   
                                    , "PeakKNNGraph1.png":convertimg(PeakKNNGraph1) 
                                    , "PeakKNNGraph2.png":convertimg(PeakKNNGraph2)  
                                    , "FEL2D1.png":convertimg(FEL2D1)    
                                    , "FEL2D2.png":convertimg(FEL2D2)    
                                    , "PATH.png":convertimg(PATH)  
                                    }
                        )
                                
if __name__ == "__main__":
    main()




