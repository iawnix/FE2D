import sys
import math
import pandas as pd
import numpy as np
import re
import argparse
from math import sqrt
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
import matplotlib.colors as colors
np.set_printoptions(precision=6)
import heapq
import math
import rich
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from datetime import datetime
from rich.text import Text
from rich.prompt import Prompt
from rich.panel import Panel
from rich.progress import track
from rich.layout import Layout
from rich.tree import Tree
from rich.status import Status
from rich.rule import Rule
import subprocess 
import sys
from matplotlib.font_manager import FontProperties


# 初始化数据
class Dat2D():
    def __init__(self, frame, x, y, t) -> None:
        self.frame = frame
        self.x = x
        self.y = y
        self.time = t

class MD2DDat():
    def __init__(self, fp, dt) -> None:
        self.frame = None
        self.x = None
        self.y = None
        self.time = None
        self.__ReadDat__(fp)
        self.__TimeSeries__(dt)

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
    
    # 此方法用于暴露类中的数据
    def values(self):
        var = Dat2D(self.frame, self.x, self.y, self.time)
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

        table = Table()
        table.add_column("[yellow]Area of window")
        table.add_column("[blue]Total Numbers of point")
        table.add_column("")
        table.add_column("[yellow]Original")
        table.add_column("[blue]Normalization")
        table.add_row("{:.2f}".format(self.mesh2d.s), "{}".format(N), "Maximum Value", "{:.4f}".format(old_max), "{:.4f}".format(new_max))
        table.add_row(""                            , ""            , "Minimum value", "{:.4f}".format(old_min), "{:.4f}".format(new_min))
        rprint(table)


# 峰值点
# 峰值点部分
# 注意，del需要在计算G之后调用，切记
class PPeak():
    def __init__(self,n_peak,mesh2d) -> None:
        self.x = []
        self.y = []
        self.mesh2d = mesh2d
        #self.ij = np.flip(peak_local_max(self.mesh2d.Pmt,num_peaks=n_peak),axis =1)
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
# 用于从终端获取哪些峰值点需要被删除

def InPeakIndex(ss):
    ss = input("Index: ")
    s = ss.rstrip("\n").split(",")
    out = [eval(i) for i in s]
    return out

# 权重graph可以复用
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

# 峰值点聚类KNN

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
        distribution_max_i = np.argmax(self.distribution_y)
        tree = Tree("{}: ({:.2f},{:.2f})".format(peak,x,y))
        Frametree = tree.add("Frame")
        Frametree.add("Maximum value: {}".format(int(self.max_)))
        Frametree.add("Minimum value: {}".format(int(self.min_)))
        Frametree.add("Average value: {:.2f}".format(self.ave_))
        Frametree.add("Median value: {:.2f}".format(self.med_))
        Frametree.add("Maximum probability distribution: {:.2f}".format(self.distribution_x[distribution_max_i]))
        Frametree.add("Distribution count: {}".format(int(self.distribution_y[distribution_max_i])))
        if self.DatTime != None:
            Timetree = tree.add("Time (ns)")
            Timetree.add("Maximum value: {:.2f}".format(int(self.DatTime[1])))
            Timetree.add("Minimum value: {:.2f}".format(int(self.DatTime[0])))
            Timetree.add("Average value: {:.2f}".format(self.DatTime[2]))
            Timetree.add("Median value: {:.2f}".format(self.DatTime[3]))
            Timetree.add("Maximum probability distribution: {:.2f}".format(self.DatTime[4][distribution_max_i]))
            Timetree.add("Distribution count: {}".format(int(self.DatTime[5][distribution_max_i])))
        rprint(tree)

class PeakCluster():
    def __init__(self, mddat2d) -> None:
        self.mddat2d = mddat2d
        self.KNNdata = None
        
    def KNN(self, peak, cutoff):
        label = []
        cutbool = []
        xpeak = peak.x
        ypeak = peak.y

        x_sample = self.mddat2d.x
        y_sample = self.mddat2d.y
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


    def KNNrep(self, outname, nbins):
        dic = {}
        # 读取KNNdata
        PeakSet = sorted(set(self.KNNdata["label"]))
        with open(outname, "w+") as F:
            for p in PeakSet:
                p_frame = self.KNNdata[self.KNNdata["label"] == p]["frame"].to_list()
                F.write("P{}:\n".format(p))
                var  = [str(j) for j in p_frame]
                F.writelines(",".join(var)+"\n")
                F.writelines("END\n")
                dic[str(p)] = self.__rep__(p_frame, nbins)
        return dic

    # 输出每个构象微状态的代表构象信息
    def Prep(self,dic,peak,dt):
        rep_keys = sorted(dic.keys())
        for i in rep_keys:
            var = dic[i]
            var.Frame2Time(dt)
            var.Pinfo("P{}".format(i), peak.x[eval(i)],peak.y[eval(i)])

    def KNNtransfer(self, T):

        dat = self.KNNdata.copy(deep=True)
        # 构建转移矩阵
        #    - Cmatrix: 计数矩阵
        #    - Tmatrix: 转移概率矩阵
        n_state = len(set(dat["label"].to_list()))
        Cmatrix = np.zeros(shape=(n_state, n_state))
        n_tran = 0
        rprint("Calculation of Transition Matrix between States")
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
        rprint(TranferMatrax)
        #out_ss = ""
        #shape_x, shape_y = TranferMatrax.shape
        #for i in range(shape_x):
        #    for j in range(shape_y):
        #        P_i2j = TranferMatrax.iloc[i,j]
        #        out_ss += "P(state[{}]->state[{}])={:.2f}%\t".format(i,j,P_i2j*100)
        #        if j == (shape_y-1):
        #            out_ss += "\n"
        #return out_ss

# 自由能计算模块
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

    # 提取Peak中的峰值点的G
    ij = Peak.ij
    g_peak = []
    for i in range(ij.shape[0]):
        g_peak.append(G[ij[i,0],ij[i,1]])
    
    table = Table()
    table.add_column("")
    table.add_column("Original Max Value")
    table.add_column("Original Min Value")
    table.add_column("Transformed Value")
    table.add_column("Transformed Max Value")
    table.add_column("Transformed Min Value")
    
    table.add_row("Free Energy"
                  ,"{:.4f}".format(G_max)
                  ,"{:.4f}".format(G_min)
                  ,"{:.4f}".format(G_max + (G_max-G_min)/10)
                  ,"{:.4f}".format(G.max())
                  ,"{:.4f}".format(G.min()))

    rprint(table)
    return G,g_peak


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
        rprint("[bold yellow]4. Griddate interpolation Module")
        #rprint("The data will be saved as \"Gbbis-Griddate.csv\"")
        old_max = self.Gbbis.flatten().max()
        old_min = self.Gbbis.flatten().min()
        #rprint("Gibbs_MAX={:.2f},Gibbs_MIN={:.2f}".format(old_max, old_min))
        X,Y,Z = self.__IntNum__(xnum, ynum)
        new_max = self.Gbbis.flatten().max()
        new_min = self.Gbbis.flatten().min()
        table = Table()
        table.add_column("")
        table.add_column("[yellow]Original Maximum value")
        table.add_column("[blue]Original Minimum value")
        table.add_column("[yellow]Interpolation by Griddata Maximum value")
        table.add_column("[blue]Interpolation by Griddata Minimum value")
        table.add_row("Free Energy",
                      "{:.2f}".format(old_max)
                      ,"{:.2f}".format(old_min)
                      ,"{:.2f}".format(new_max)
                      ,"{:.2f}".format(new_min))
        rprint(table)
        pd.DataFrame([X.flatten(),Y.flatten(),Z.flatten()]).T.to_csv("Gbbis-Griddate.csv")
        return X,Y,Z
    
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

def RunTermGraph(cmd:str, IN_: str) -> str:
    ret1 = subprocess.Popen(cmd
                            ,bufsize=-1
                            ,shell=False
                            ,encoding="utf-8"
                            ,stdin=subprocess.PIPE
                            ,stdout=subprocess.PIPE
                            ,stderr=subprocess.PIPE)
 
    ret1_ = ret1.communicate(input= IN_)
    out1,error1 = ret1_[0],ret1_[1]
    code = ret1.returncode

    if error1 != "":
        if not code:
            return out1[1:].rstrip("\n")
        else:
            rprint("Error: {}".format(error1))
            sys.exit(1)
    else:
        return out1[1:].rstrip("\n")

def Draw2DPeakProbability(mddat2d, xnum, ynum):
    xout = ""
    xhist, xbins = np.histogram(mddat2d.x, bins = xnum, density= False)
    for i, i_t in enumerate(xbins[:-1]):
        x_ = (xbins[i+1] + i_t)/2
        xout += "{:.2f},{:.2f}\n".format(x_,xhist[i])
    
    yout = ""
    yhist, ybins = np.histogram(mddat2d.y, bins = ynum, density= False)
    for i, i_t in enumerate(ybins[:-1]):
        y_ = (ybins[i+1] + i_t)/2
        yout+= "{:.2f},{:.2f}\n".format(y_,yhist[i])
    # 储存数据
    with open("X-Distribution.csv", "w+") as F:
        F.writelines(xout)
    with open("Y-Distribution.csv", "w+") as F:
        F.writelines(yout)
   
   
    return xout, yout

def Header(console):
    helloW = '''
_____ _____ ____  ____    _             ___    ___        __ 
|  ___| ____|___ \|  _ \  | |__  _   _  |_ _|  / \ \      / /
| |_  |  _|   __) | | | | | '_ \| | | |  | |  / _ \ \ /\ / / 
|  _| | |___ / __/| |_| | | |_) | |_| |  | | / ___ \ V  V /  
|_|   |_____|_____|____/  |_.__/ \__, | |___/_/   \_\_/\_/   
                                 |___/                       
Welcome to use FE2D by IAW[HENU], if you have any question, please ask PhD Zhang Zhiyang and M.A. Chen Juyuan
'''

    console.print(Text(helloW, justify = "left", style = "bold green"))



# 有机会可以做好图片字体的渲染
plt.switch_backend("agg")
plt.rcParams["axes.labelweight"] ="bold"
plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.weight"]="bold"
plt.rcParams["font.size"]=20

## 用于绘制散点-时间分布
def draw2DScatterTime(md2ddat,xlabel,ylabel,outname):
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)

    ax.set_xlabel("{}".format(xlabel))
    ax.set_ylabel("{}".format(ylabel))
    ax.set_xlim(md2ddat.x.min(),md2ddat.x.max())
    ax.set_ylim(md2ddat.y.min(),md2ddat.y.max())
    plt.scatter(md2ddat.x,md2ddat.y,alpha = 0.7,s= 5,c=md2ddat.time)
    cbar = plt.colorbar()
    cbar.set_label('Time (ns)')
    plt.tight_layout()
    plt.savefig("{}.tiff".format(outname))
    plt.savefig("{}.jpg".format(outname))

def draw2DScatterCircle(md2ddat,slabel,myPeak, cutoff, outname,xlabel,ylabel):
    
    dat = pd.DataFrame({"x":md2ddat.x,"y":md2ddat.y,"label":slabel})
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)

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
    plt.savefig("{}.tiff".format(outname))
    plt.savefig("{}.jpg".format(outname))

def draw2DFEL(X, Y, Z, cname ,select_plot, xlabel, ylabel, outname):
    # 绘制概率密度曲线
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)

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
    plt.savefig("{}.tiff".format(outname))
    plt.savefig("{}.jpg".format(outname))


def drawPeakTimeDistribution(dic,dt, xlabel, ylabel, outname):
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)
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
    plt.savefig("{}.tiff".format(outname))
    plt.savefig("{}.jpg".format(outname)) 

        


# 这里有一个Bug，我们将极值点直接映射到PATH上了
def drawPATH(X,Y,Z,cname,traj_x, traj_y,select_plot,xlabel, ylabel,outname):
    # 绘制概率密度曲线
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(4)
    ax.spines['top'].set_linewidth(4)
    ax.spines['left'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4)

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
    plt.savefig("{}.tiff".format(outname))
    plt.savefig("{}.jpg".format(outname))

def Parm():
    parser = argparse.ArgumentParser(description=
                                     "The author is very lazy and doesn't want to write anything\n"
                                     "Author: ZJH [HENU]"
                                    )
    parser.add_argument("-Fp",type=str, nargs=1, help="FilePath")
    parser.add_argument("-dimX",type=str, nargs=1, help="X Number of Sample Windows")
    parser.add_argument("-dimY",type=str, nargs=1, help="Y Number of Sample Windows")
    parser.add_argument("-SmoX",type=str, nargs=1, help="Smooth num of X")
    parser.add_argument("-SmoY",type=str, nargs=1, help="Smooth num of Y")
    #parser.add_argument("-nbinF",type=str, nargs=1, help="number of bins Feature")
    parser.add_argument("-nbinT",type=str, nargs=1, help="number of bins Time")
    parser.add_argument("-T",type=str, nargs=1, help="Temperature (K)")
    parser.add_argument("-Cutoff",type=str, nargs=1, help="???")
    parser.add_argument("-Pn",type=str, nargs=1, help="The num of Peaks")
    parser.add_argument("-dt",type=str, nargs=1, help="Time scale")
    parser.add_argument("-Cmp",type=str, nargs=1, help="Color")
    parser.add_argument("-Xlab",type=str, nargs=1, help="Xlabel")
    parser.add_argument("-Ylab",type=str, nargs=1, help="Ylabel")
    return parser.parse_args()

def main():
    parg = Parm()

    fp = parg.Fp[0]
    xnum = eval(parg.dimX[0])      
    ynum = eval(parg.dimY[0])      
    xintN = eval(parg.SmoX[0])   
    yintN = eval(parg.SmoX[0])   
    dt = eval(parg.dt[0])
    n_peak = eval(parg.Pn[0])     
    cutoff = eval(parg.Cutoff[0]) 
    #nbinF = eval(parg.nbinF[0])
    nbinT = eval(parg.nbinT[0])
    T = eval(parg.T[0])
    # 定义绘图的cmap 
    mycmp = parg.Cmp[0]           
    xlabel = parg.Xlab[0]
    ylabel = parg.Ylab[0]
    
    # 初始化用于终端渲染的
    myconsole = Console(style= None)
    Header(myconsole)
    rprint(Rule())
    rprint("")

    mymd = MD2DDat(fp,dt)
    # 绘制两组变量在1D水平上的分布
    rprint("[bold]1. Data Initialization Module")
    rprint("1.1 Probability Distribution of input data")
    Probability_x, Probability_y = Draw2DPeakProbability(mymd,xnum, ynum)
    rprint("{}".format(xlabel))
    rprint(RunTermGraph("termgraph", Probability_x))
    rprint("{}".format(ylabel))
    rprint(RunTermGraph("termgraph", Probability_y))
    
    rprint("1.2 Time Distribution")
    with Status("[bold red]Generate Image...[/]"):
        draw2DScatterTime(mymd,xlabel,ylabel,"Time-Distribution")
        rprint("Successful Distribution Map!\n")
    myP = GenPMatrix2D(mymd, xnum, ynum)
    myP.Pmatrix()
    rprint(Rule())
    rprint("")

    rprint("[bold]2. Peak Analysis Module")
    myPeak = PPeak(n_peak,myP.mesh2d)
    rprint("[cyan bold]Probability Density [red bold]----------> [cyan blod]Free Energy")
    myG = CalGibbs(myP.mesh2d.Pmt, myPeak, T)
    myPeak.setG(myG[1])
    
    # 这里需要绘制第一张聚类后的散点图用于
    myPeakKNN_test = PeakCluster(mymd)
    # 进行KNN聚类
    myKnnTestLabel,myKnnTestCut = myPeakKNN_test.KNN(myPeak, cutoff)
    # 输出采样代表构象信息
    myrepTest = myPeakKNN_test.KNNrep("./FreeEnergy-Peak.txt",nbinT)
    myPeakKNN_test.Prep(myrepTest,myPeak, dt)
    with Status("[bold red]Generate Image...[/]"):
        draw2DScatterCircle(mymd,myPeakKNN_test.KNNdata["label"].to_list(),myPeak, cutoff, "Time-Distribution-withPeak",xlabel,ylabel)
        rprint("Successful: Time-Distribution-withPeak")

    del_point_i = Prompt.ask(Text("Enter the point you want to delete(\"-1\": Nonething)", style = "bold red"))
    if del_point_i != "-1":
        del_point_list = [eval(i) for i in del_point_i.rstrip("\n").split(",")]
        # 删除去除的peak
        myPeak.delPeak(del_point_list)
        myPeakKNN = PeakCluster(mymd)
        myKnnLabel,myKnnCut = myPeakKNN.KNN(myPeak, cutoff)
        myrep = myPeakKNN.KNNrep("./FreeEnergy-Peak.txt",nbinT)
        myPeakKNN.Prep(myrep,myPeak, dt)
        with Status("[bold red]Generate Image...[/]"):
            draw2DScatterCircle(mymd,myPeakKNN.KNNdata["label"].to_list(),myPeak, cutoff, "Time-Distribution-withPeak",xlabel,ylabel)
            rprint("Successful: Time-Distribution-withPeak")
    else:
        myKnnLabel, myKnnCut = myKnnTestLabel, myKnnTestCut
        myrep = myrepTest
        myPeakKNN = myPeakKNN_test

    with Status("[bold red]Generate Image...[/]"):
    # 需要绘制Peak在时间上的分布
        drawPeakTimeDistribution(myrep, dt,"Time (ns)", "Count", "Peak-Time-Distribution")
        rprint("Successful: Peak Time Distribution")

    with Status("[bold red]Generate Image...[/]"):
        draw2DFEL(myP.mesh2d.x
                  , myP.mesh2d.y
                  , myG[0].T
                  , mycmp
                  ,(myPeak.x, myPeak.y)
                  ,xlabel
                  ,ylabel
                  ,"FreeEnergy")
        rprint("Successful original 2D-FEL")
    rprint(Rule())
    rprint("")

    rprint("[bold]3. Peak Transfer and Path Module")
    myTmatrax = myPeakKNN.KNNtransfer(T = int(len(mymd.x)/nbinT))
    myPeakKNN.Ptransfer(myTmatrax)
    # 开始进行PATH分析
    start_end_s = Prompt.ask(Text("Enter the points you want to transfer(e.g. 0-1: P0 -> P1)", style = "bold red"))
    start_i, end_i = start_end_s.rstrip("\n").split("-")
    start = "{}_{}".format(myPeak.ij[eval(start_i),0], myPeak.ij[eval(start_i),1])
    end =  "{}_{}".format(myPeak.ij[eval(end_i),0], myPeak.ij[eval(end_i),1])

    mypath = PeakPATH(myG[0])
    
    par,dis = mypath.dijkstra(start)
    traj = mypath.Traj(start, end, par)
    trajdis = mypath.TrajDis(traj)

    # 进行插值算法       
    myFE = GdDInter(myP.mesh2d, myG[0])
    # 插值
    EXintN,FEYintN, FEZintN = myFE.run(xintN, yintN)
    # 寻找峰值点
    FEpeakFind = myFE.FindPeak(myPeak.x, myPeak.y, EXintN.flatten(), FEYintN.flatten())
    select_plot_x, select_plot_y = [], []
    for i in sorted([eval(i) for i in FEpeakFind.keys()]):
        select_plot_x.append(FEpeakFind[str(i)][0])
        select_plot_y.append(FEpeakFind[str(i)][1])
    
    with Status("[bold red]Generate Image...[/]"):
        draw2DFEL(EXintN
                  ,FEYintN
                  ,FEZintN.T
                  , mycmp
                  , (select_plot_x, select_plot_y)
                  ,xlabel
                  ,ylabel
                  ,"FreeEnergy-InterNum")
        rprint("Successful: FreeEnergy-InterNum")
    

    # 寻找PATH的点
    PathPoint = myFE.FindPeak(myP.mesh2d.x[np.array(traj)[:,0]]
                             , myP.mesh2d.y[np.array(traj)[:,1]]
                             , EXintN.flatten()
                             , FEYintN.flatten())
    traj_x, traj_y = [], []
    for i in sorted([eval(i) for i in PathPoint.keys()]):
        traj_x.append(PathPoint[str(i)][0])
        traj_y.append(PathPoint[str(i)][1])

    with Status("[bold red]Generate Image...[/]"):
        drawPATH(EXintN, FEYintN, FEZintN.T
                 ,mycmp 
                 ,traj_x ,traj_y
                 ,(select_plot_x, select_plot_y)
                 ,xlabel, ylabel
                 ,"FreeEnergy-PATH")
        rprint("Successful: FreeEnergy-PATH")

if __name__ == "__main__":
    main()
