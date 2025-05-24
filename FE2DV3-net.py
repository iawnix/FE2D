import streamlit as st
import pandas as pd
import numpy as np
import sys
import math
from scipy import interpolate
from scipy.interpolate import RBFInterpolator
from scipy.interpolate import Rbf
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from matplotlib import pyplot as plt
import matplotlib.colors as colors
from io import BytesIO

@st.cache_data
def load_data(fp):
    print("Load Data from :{}".format(fp))
    data = pd.read_csv(fp, sep = "," ,header= None)
    return data

def TimeSeries(Frames, dt):
    t = {"ps":[],"ns":[]}
    for i in Frames:
        t["ps"].append(i*dt/1000)
        t["ns"].append(i*dt)
    return t


# 以下是概率密度计算
## 该函数用于生成网格中心点，df: 读取的DataFrame, dnum : 采样网格维度 dnum*dnum
def centerPot(df,dnum):
    x_min,x_max = df[1].min(), df[1].max()
    y_min,y_max = df[2].min(), df[2].max()
    dx = (x_max-x_min) / (2*dnum)
    dy = (y_max-y_min) / (2*dnum)
    x_center = np.array([])
    y_center = np.array([])
    #print("dx = {}, dy = {}".format(dx,dy))
    func = lambda x,d,c0 : (2*x-1)*d + c0
    # 网格面积
    s = 2*dx*2*dy           
    for i in range(1,dnum+1):
        x_center = np.append(x_center,func(i,dx,x_min))
        y_center = np.append(y_center,func(i,dy,y_min))
        #print("x = {} , y  = {} , dx_ = {} , dy_ = {}".format(func(i,dx,x_min),func(i,dy,y_min),(2*i-1)*dx,(2*i-1)*dy))
    return (x_center,y_center,s,dx,dy)

## 该函数用于落在网格中心点的采样点数
# DataSet: list, [Frame,Pca1,Pca]
def CloseWPot(DataSet,center,dx,dy):
    F,x,y = DataSet
    newx = []
    newy = []
    newf = []
    # 用于储存落入网格中的frame信息
    frame = []
    count = 0
    # 遍历寻找落在当前窗格的数据点个数
    for i in range(len(x)):
        if (center[0] - dx <= x[i] <   center[0] + dx) and (center[1] - dy <= y[i] <   center[1] + dy) :
            count += 1
            # 添加符合要求的frame
            frame.append(F[i])
        # 删除统计过的数据点
        else:
            newx.append(x[i])
            newy.append(y[i])
            newf.append(F[i])
    return ([newf,newx,newy],count,frame)

## 用于进行每个窗口的概率密度计算 df: 读取的DataFrame, dnum : 采样网格维度 dnum*dnum
@st.cache_data
def CountW(df,dnum):

    fxy = [df[0],df[1],df[2]]
    # 总采样点
    N = len(fxy[0])
    cPout = centerPot(df,dnum)
    
    # 生成的中心网格点是一个方阵
    x_center,y_center,s,dx,dy = cPout
    z_center = np.zeros([len(x_center),len(y_center)])

    # 创建dict储存采样网格原始frame信息，其中keys的格式为"i-j"
    f_center = {}
    
    print("")
    print("Ready to cal P")
    print("S_window={:.2f}\t\tN_total={}".format(s,N))
    
    for i in range(len(x_center)):
        for j in range(len(y_center)):
            cwPout = CloseWPot(DataSet = fxy
                                        ,center = (x_center[i],y_center[j])
                                        ,dx = dx
                                        ,dy = dy
                                    )
            # 更新fxy
            fxy = cwPout[0]
            # 计算该点的概率密度
            P = cwPout[1] / (N*s)
            # 可能会出现大于1的情况
            if P > 1:
                print("Warning P > 1")
                print("P({},{})={}\nx={},y={}\ncount={},N={},s={}".format(i,j,P,x_center[i],y_center[j],cwPout[1],N,s))
            # 记录数值
            z_center[i][j] = P
            # 因为下一步转至了，所以这一步是j i
            f_center["{}-{}".format(j,i)] = cwPout[2]
    # 最后生成的z需要转置
    z_center = z_center.T
    print("Pmax={},Pmin={}".format(z_center.max(),z_center.min()))
    # 归一化
    tol_density = np.sum(z_center)
    z_center /=tol_density
    print("After Normalization, Pmax={},Pmin={}".format(z_center.max(),z_center.min()))
    print("")
    return (x_center,y_center,z_center,f_center)



# 以下是绘图部分
## 全局字体设置
plt.rcParams["axes.labelweight"] ="bold"
#plt.rcParams["font.family"]="Times New Roman"
plt.rcParams["font.weight"]="bold"
plt.rcParams["font.size"]=10

def draw2DGibbs(xyz,cname,peak):
    # 绘制概率密度曲线
    X,Y,Z = xyz
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.set_xlim(X.min(),X.max())
    ax.set_ylim(Y.min(),Y.max())
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    C=plt.contour(X,Y,Z,5,colors='black',linewidths= 0.2)  #生成等值线图
    plt.contourf(X,Y,Z,5,alpha=0.2)
    Pcolor= plt.pcolor(X, Y, Z
                       ,shading='auto',cmap=cname
                       ,norm = colors.TwoSlopeNorm(vmin=Z.min(), vcenter=(Z.max()-Z.min())/2, vmax=Z.max()))
    plt.colorbar(Pcolor)
    plt.scatter(peak.x,peak.y,color = "black",s = 6)
    st.pyplot(fig)
    return fig

## 用于绘制散点分布，以及cutoff的截断圆
def draw2DScatterCircle(df,slabel,myPeak, cutoff):
    dat = df.copy()
    dat[3] =  slabel
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    ax.set_xlim(dat[1].min(),dat[1].max())
    ax.set_ylim(dat[2].min(),dat[2].max())
    for i in set(slabel):
        Datai = dat[dat[3] == i]
        plt.scatter(Datai[1],Datai[2],alpha = 0.7,s= 5)
    # 绘制峰值点:
    for i,x_ in enumerate(myPeak.x):
        plt.scatter(x_,myPeak.y[i],color = "black")
        plt.text(x_,myPeak.y[i],s="P{}".format(i))
        # 并绘制cutoff
        draw_circle = plt.Circle((x_,myPeak.y[i]),cutoff,fill=False)
        plt.gcf().gca().add_artist(draw_circle)

    st.pyplot(fig)
    return fig
## 用于绘制散点-时间分布
def draw2DScatterTime(df,Time):
    fig,ax = plt.subplots(dpi=300)
    # 设置边框
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(df[1].min(),df[1].max())
    ax.set_ylim(df[2].min(),df[2].max())
    plt.scatter(df[1],df[2],alpha = 0.7,s= 5,c=Time["ns"])
    cbar = plt.colorbar()
    cbar.set_label('Time (ns)')
 
    st.pyplot(fig)
    return fig

## 绘制空图用于初始化界面
def drawNULL():
    fig,ax = plt.subplots(dpi=300)
    plt.axis("off")
    st.pyplot(fig)

# 峰值点部分
# 注意，del需要在计算G之后调用，切记
class PPeak():
    def __init__(self,n_peak,DATA) -> None:
        self.x = []
        self.y = []
        self.DATA = DATA
        self.ij = peak_local_max(self.DATA[2],num_peaks=n_peak)
        self.G = []
        self._xy()
    def _xy(self):
        for i in range(self.ij.shape[0]):
            self.x.append(self.DATA[0][self.ij[i,1]])
            self.y.append(self.DATA[1][self.ij[i,0]])
    # 用于删除指定索引
    def delPeak(self,i_list):
        x = []
        y = []
        G = []
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
    def _g(self,G):
        self.G = G
# 用于从终端获取哪些峰值点需要被删除
def GInDelPeakIndex(ss):
    s = ss.rstrip("\n").split(",")
    out = [eval(i) for i in s]
    return out

# 此函数用于从一系列frame中寻找一个最合适的Rep，首先进行3σ准则，排除异常值，再进行从小到大的排序，如果是偶数，取(len - 1 )//2，如果是奇数取 (len - 1) / 2
# 新增功能，用于计算去除异常值之后的Frames的平均值，这里不做取整，主要表示时间，同时返回最小值，最大值
# 定义数据包用于FindPeakRep函数
class CircleData():
    def __init__(self,rep,ave_,max_,min_):
        self.rep = rep
        self.ave = ave_
        self.ma = max_
        self.mi = min_        

def FindPeakRep(Frames):
    new = []
    exceptnum = []
    mean_ = sum(Frames) / len(Frames)
    std = np.std(np.array(Frames))
    up = mean_ + 3*std
    limit = mean_ - 3*std
    for i,x in enumerate(Frames):
        if (x >  up) or (x <  limit):
            exceptnum.append(x)
        else:
            new.append(x)
    if (len(new) - 1)%2 != 0:
        var = len(new) - 2          # 偶数个
    else:
        var = len(new) - 1          # 奇数个

    rep = new[int(var/2)]
    ave = sum(new)/len(new)
    min_ = min(new)
    max_ = max(new)
    myCircleData = CircleData(rep,ave,max_,min_)
    return myCircleData

## 采用KNN分类用于对峰值周围的点进行分类
### ct_x : 储存中心点的x坐标 ct_y ：储存中心点的y坐标 DATA: 储存待分类数据，数据结构为pandas
### 20230317：
### 搞明白当初引入cutoff的意义，只能说我太NB了，用于只保留峰值点周围多少距离的数据，应保持较小的cutoff
### ，使落入cutoff的点距离极值点比较近，且计算中位数，方便提取构象

def KNN(ct_x,ct_y,DATA,cutoff):
    label = []
    dist = []
    for i in range(DATA.shape[0]):
        fi = DATA.iloc[i,0]
        xi = DATA.iloc[i,1]
        yi = DATA.iloc[i,2]
        d = {}
        Fd = lambda x0,y0,x1,y1: math.sqrt(math.pow(x1 - x0, 2) + math.pow(y1 - y0, 2)) 
        for i_,x_ in enumerate(ct_x):
            d[str(i_)] = Fd(x_,ct_y[i_],xi,yi)
        d_ = sorted(d.items(),  key=lambda d: d[1], reverse=False)[0]
        labeli = eval(d_[0])
        label.append(labeli)
        disti = d_[1]
        if disti <= cutoff:
            dist.append(1)
        else:
            dist.append(0)
        
    return (label,dist)
        
## 此函数用于计算KNN聚类中，每隔cutoff circle中的数据frame信息
## DATA 最原始数据，lable KNN返回的， cutbool: KNN返回，标志是否位于阶段之中
def CTinfo(DATA, label, cutbool):
    out = []
    Data = DATA.copy()
    Data[3] = label
    Data[4] = cutbool
    for i in set(label):
        out.append(FindPeakRep(list(Data.loc[(Data[3] == i) & (Data[4] == 1),:][0])))
    return out
# 此函数用于返回KNN聚类中，每个cutoff circle中的数据frame信息
def CTframe(DATA, label,cutbool):
    out = {}
    Data = DATA.copy()
    Data[3] = label
    Data[4] = cutbool
    for i in set(label):
        out[str(i)] = list(Data.loc[(Data[3] == i) & (Data[4] == 1),:][0])
    return out


## 定义函数用于输出CTinfo的信息
def PrintCTinfo(CTinfo_out, mypeak,dt):
    out = ""
    for i,x_ in enumerate(mypeak.x):
        out += ("P{}:\n".format(i))
        out += ("Frame: Rep = {}, Ave = {}, Range: {} -> {}\n".format(CTinfo_out[i].rep,CTinfo_out[i].ave,CTinfo_out[i].mi,CTinfo_out[i].ma))
        out += ("Time: Ave = {:.2f} (ns), Range: {:.2f} -> {:.2f} (ns)\n".format(CTinfo_out[i].ave*dt,CTinfo_out[i].mi*dt,CTinfo_out[i].ma*dt))
        out += ("Gibbs = {:.2f}\n".format(mypeak.G[i]))
        out += ("\n")
    return out

# 定义函数用于输出CTframe信息
# 因为frame是从1开始，所以输出索引的时候直接-1就好，也就是需要一个参数sF：
def PrintCTframe(CTframe_out, sF,outf):
    with open(outf, "w+") as F:
        for i in CTframe_out.keys():
            F.writelines("P{}:\n".format(i))
            var = [ str(f - sF) for f in CTframe_out[i]]
            F.writelines(",".join(var)+"\n")
            F.writelines("END\n".format(i))

# 自由能计算模块
# 该函数用于计算相对的Gibbs
# 修订, 新增计算峰值点的Gibble
# 修订, G-G_min之后重新计算max,min用于去填补nan
def CalGibbs(P_martix, Peak, T):                        # 20230426: 之前这一块貌似一直用的绝对零度273.15
    k = 1.3806505*10**(-23)
    C = 0 
    c1 = 4186
    NA = 6.023*10**(23)                                 # 20230426: SB ......... kT ->kcal/mol
    # flatten
    shape = P_martix.shape
    P_ = P_martix.flatten()
    
    
    # 概率密度为0的密度点的Gibbs会直接变为0,P < 1，所以
    Gibbs = lambda P: -np.nan if abs(P - 0) <= 1e-10  else ((-k*T*np.log(P) + C)/c1)*NA
    G = np.array(list(map(Gibbs,P_))).reshape(shape)
    var = G[~np.isnan(G)]
    G_min = var.min()
    G_max = var.max()
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
 
    print("G_max={:.2f},G_min={:.2f},GnoSample(Nan)={:.2f}".format(G_min,G_max,G_max + (G_max-G_min)/10))
    print("")
    return G,g_peak

## 此函数用于插值
# 采用griddata进行插值
def InterNUM_ggd(XX,YY,Z,x,y,Gridnum):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    xx,yy = np.meshgrid(np.linspace(xmin,xmax,num = Gridnum,endpoint=True)
                        ,np.linspace(ymin,ymax,num = Gridnum,endpoint=True))
    newz = interpolate.griddata((XX.flatten()
                                 ,YY.flatten())
                                 ,Z.flatten(),(xx,yy)
                                 , method='linear'
                                 ,fill_value =0)

    return (xx,yy,newz)

def G_InterNUM_ggd(myP,Gridnum,Gibbs):
    # 根据网格密度生成网格
    X,Y = np.meshgrid(myP[0],myP[1])
    Z = Gibbs
    # 进行GIBBS计算
    print("Gibbs_MAX={:.2f},Gibbs_MIN={:.2f}".format(Z.max(),Z.min()))
    print("Gibbs InterNUM_ggd:")
    X,Y,Z = InterNUM_ggd(X,Y,Z,myP[0],myP[1],Gridnum)
    print("Gibbs_MAX={:.2f},Gibbs_MIN={:.2f}".format(Z.max(),Z.min()))
    #c
    return (X,Y,Z)

def convertimg(fig):
    buf = BytesIO()
    fig.savefig(buf,format="PNG")
    buf.seek(0)
    return buf.getvalue()


def downDATA(colnum,Gbbis,Gbbis_inNum,PCAT, KNMPEAK,GBBISF):
    Gbbis = pd.DataFrame(Gbbis).to_csv()
    Gbbis_gdd = pd.DataFrame([Gbbis_inNum[0].flatten(),Gbbis_inNum[1].flatten()
                              ,Gbbis_inNum[2].flatten()]).T.to_csv()
    
    # streamlit设置数据下载按钮
    with colnum[0]:
        st.download_button(
            label='Download',
            key='d1',
            data=Gbbis,
            file_name='Gibbs.csv',
            mime='text/csv',
            use_container_width = True
        )
    with colnum[1]:
        st.download_button(
            label='Download',
            data=Gbbis_gdd,
            key='d2',
            file_name='Gibbs_gdd.csv',
            mime='text/csv',
            use_container_width = True
        )
    with colnum[2]:
        pcatfig = convertimg(PCAT)
        st.download_button(
            label="Download",
            data=pcatfig,
            key='d3',
            file_name="PCAT.png",
            mime="image/png",
            use_container_width = True
            )
    with colnum[3]:  
        KNMPEAKfig = convertimg(KNMPEAK)
        st.download_button(
            label="Download",
            data=KNMPEAKfig,
            key='d4',
            file_name="KNMPEAK.png",
            mime="image/png",
            use_container_width = True
            )
    with colnum[4]:
        GBBISfig = convertimg(GBBISF)
        st.download_button(
            label="Download",
            data=GBBISfig,
            key='d5',
            file_name="GIBBS.png",
            mime="image/png",
            use_container_width = True
            )

def printweb(colnum,ssname,txt):
    with colnum:
        with st.session_state[ssname]:
            st.code(txt, language='python')


def main():

    st.set_page_config(
        page_title = "FE2D By IAW"
        ,layout = "wide"
    )

    st.title('FE2D')

    # 采用表单，固定参数写入到侧边栏

    with st.sidebar.form('my_form'):
        col5, col6 = st.columns(2)
        with col5:
            dnum = st.number_input("Shape", format='%d',step = 5)
            dnum = int(dnum)
        with col6:
            Gridnum = st.number_input("Grid", format='%d',step = 5)
            Gridnum = int(Gridnum)
        col7, col8 = st.columns(2)
        with col7:
            T = st.number_input("T", format='%d',step = 5)
        with col8:
            dt = st.number_input("dt",step = 0.02)
        cutoff = st.number_input("Cutoff",step = 0.5)
        npeak = st.number_input("Npeak", format='%d',step = 1)
        fp = st.file_uploader("Data", type="csv")
        submitted = st.form_submit_button('Submit')
    
    col1, col2 = st.columns(2)
    
    ss_del_peak = st.text_input("Peak index to be deleted: ")
    col3, col4 = st.columns(2)
    coldown,colout = st.columns(2)
    global image1,image2,image3,image4,stdOUT,txt,PCAT, KNMPEAK,GBBISF

    # 用于记录所有的原print内容
    st.session_state['txt'] = "" 
    with col1:
        st.write("Distribution-Time Graph")
        st.session_state['image1'] = st.empty()
        with st.session_state['image1']:
            drawNULL()      
    with col2:
        st.write("All Peaks Graph")
        st.session_state['image2'] = st.empty()
        with st.session_state['image2']:
            drawNULL()    
    with col3:
        st.write("Selected Peak Graph")
        st.session_state['image3'] = st.empty()
        with st.session_state['image3']:
            drawNULL()    
    with col4:
        st.write("Gibbs Graph")
        st.session_state['image4'] = st.empty()
        with st.session_state['image4']:
            drawNULL()  
    with coldown:
        st.write("Data download")
        downtab = st.tabs(["Gibbs.csv", "Gibbs-gdd.csv","Pca-T.png","Peak.png","Gibbs.png"])
  
    with colout:
        st.write("Program Run Log")
        st.session_state['stdOUT'] = st.empty()
        with st.session_state['stdOUT']:
            st.code(" ", language='python')
    #if submitted:

    st.session_state['txt'] += "FE2D is initialized successfully"
    printweb(colout,"stdOUT",st.session_state['txt'])

    if fp:
        dat = load_data(fp)
        time = TimeSeries(dat[0],dt)
        with col1:
            with st.session_state['image1']:
                PCAT = draw2DScatterTime(dat,time)

        myP = CountW(dat,dnum)
        myPeak = PPeak(npeak,myP[:3])
        st.session_state['txt'] += "Probability density calculation completed.\n"
        st.session_state['txt'] += "Pmax={},Pmin={}\n".format(myP[2].max(),myP[2].min())
        printweb(colout,"stdOUT",st.session_state['txt'])
        slabel_0,scut_0 = KNN(myPeak.x,myPeak.y,dat,cutoff)

        with col2:
            with st.session_state['image2']:
                draw2DScatterCircle(dat,slabel_0,myPeak,cutoff)
        
        if ss_del_peak:
            
            peak_del_index = GInDelPeakIndex(ss_del_peak)

            Gibbs,g_peak = CalGibbs(myP[2],myPeak,T)
            myPeak._g(g_peak)
            if ss_del_peak != -1:
                myPeak.delPeak(peak_del_index)
                st.session_state['txt'] += "Peak_Del_Index_List:\n"
                st.session_state['txt'] += ss_del_peak
                st.session_state['txt'] += "\n"
                printweb(colout,"stdOUT",st.session_state['txt'])

            slabel_1,scut_1 = KNN(myPeak.x,myPeak.y,dat,cutoff)
            
            with col3:
                with st.session_state['image3']:
                    KNMPEAK = draw2DScatterCircle(dat,slabel_1,myPeak,cutoff)
            G_ggd = G_InterNUM_ggd(myP,Gridnum,Gibbs)
            with col4:
                with st.session_state['image4']:
                    GBBISF = draw2DGibbs(G_ggd,"rainbow",myPeak)

            peakcircleinfo = CTinfo(dat, slabel_1, scut_1)
            st.session_state['txt'] += PrintCTinfo(peakcircleinfo,myPeak,dt)
            printweb(colout,"stdOUT",st.session_state['txt'])

            downDATA(downtab,Gibbs,G_ggd,PCAT, KNMPEAK,GBBISF)

if __name__ == "__main__":
    main()
