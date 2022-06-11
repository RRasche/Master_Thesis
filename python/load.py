import numpy as np
import matplotlib.pyplot as py
from matplotlib.widgets import Slider
from matplotlib import animation, rc
import matplotlib 

matplotlib.rcParams["figure.dpi"] = 300
size=14
ticksize = 14
legendsize=15
py.rc('font', size=size) #controls default text size
py.rc('axes', titlesize=size) #fontsize of the title
py.rc('axes', labelsize=size) #fontsize of the x and y labels
py.rc('xtick', labelsize=ticksize) #fontsize of the x tick labels
py.rc('ytick', labelsize=ticksize) #fontsize of the y tick labels
py.rc('legend', fontsize=legendsize) #fontsize of the legend

py.rcParams['text.usetex'] = True
preamb = r"\usepackage{bm} \usepackage{mathtools}"
params = {'text.latex.preamble' : preamb }
py.rcParams.update(params)
py.rcParams['font.family'] = "computer modern roman"




def sin(x, t, w = 1/50):
    w *= 2 * np.pi
    return np.sin(w/4 * x - w * t)

def disp1(kx, delta):
    a = 10j
    b = 3j * np.exp(-1j * kx * delta)
    c = 18j * np.exp(1j * kx * delta) 
    d = 6j * np.exp(2j * kx * delta)  
    e = 1j * np.exp(3j * kx * delta)
    return (a + b - c + d - e)/(12 * delta)

def dispo6(kx, delta):
    a1 =  -2
    b1 =  24 * np.exp(1j * kx * delta)
    c1 =  35 * np.exp(2j * kx * delta) 
    d1 = -80 * np.exp(3j * kx * delta)  
    e1 =  30 * np.exp(4j * kx * delta)  
    f1 =  -8 * np.exp(5j * kx * delta)  
    g1 =   1 * np.exp(6j * kx * delta)
    return 1j*np.exp(-2j*kx*delta)*(a1 + b1 + c1 + d1 + e1 + f1 + g1)/(60 * delta)

def disp1negative(kx, delta):
    a = - 1
    b = + 6 * np.exp(1j * kx * delta)
    c = -18 * np.exp(2j * kx * delta) 
    d = +10 * np.exp(3j * kx * delta)  
    e = + 3 * np.exp(4j * kx * delta)
    return 1j*np.exp(-3j*kx*delta)*(a + b + c + d + e)/(12 * delta)

def disp2(kx, delta):
    a = 10j
    b = 3j * np.exp(-2j * kx * delta)
    c = 18j * np.exp(2j * kx * delta) 
    d = 6j * np.exp(4j * kx * delta)  
    e = 1j * np.exp(6j * kx * delta)
    return (a + b - c + d - e)/(24 * delta)
  
def getxIdx(res):
    n2 = int(res*2)
    xIdx = np.zeros(2*n2, dtype=int)
    
    xIdx[:n2] = np.arange(-n2,n2,2)
    xIdx[n2:] = np.arange(n2,2 * n2,1)
    
    return xIdx

def plotEzAdjust(Ez, resolution):
    
    t = np.linspace(0, 2*resolution+1, (int(np.round((2*resolution+1)*10)) + 1))
    
    fig, ax = py.subplots()
    py.subplots_adjust(left=0.25, bottom=0.20)

    x = np.arange(Ez.shape[0])

    l1, = ax.plot(x, Ez[:, 0], color='green')

    l3, = ax.plot([0, x[-1]], [1.0, 1.0])
    l3, = ax.plot([0, x[-1]], [-1.0, -1.0])
    
    n2 = Ez.shape[0]//3
    f = np.zeros((Ez.shape[0], 6))
    f[:n2, 2] = sin(np.arange(0,n2), 0, 1/resolution)
    f[:n2, 4] = -sin(np.arange(0,n2), 0, 1/resolution)
    f[n2:2*n2+1, 2] = sin(np.arange(n2, 3*n2+1, 2), 0, 1/resolution)
    f[n2:2*n2+1, 4] = -sin(np.arange(n2, 3*n2+1, 2), 0, 1/resolution)
    f[2*n2+1:, 2] = sin(np.arange(3*n2+1,4*n2), 0, 1/resolution)
    f[2*n2+1:, 4] = -sin(np.arange(3*n2+1,4*n2), 0, 1/resolution)

    l4, = ax.plot(x, f[:, 2], color='red')
        
    ax.margins(x=0)

    axcolor = 'white'
    axtime = py.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

    stime = Slider(axtime, 'Timestep', 0, Ez.shape[1] - 1, valinit=0, valstep=1)
    
   
    

    def updatePlot(val):

        ax.set_title('t = ' + str(t[val]))
        l1.set_data(x, Ez[:, val])
        
        f = np.zeros((Ez.shape[0], 6))
        f[:n2, 2] = sin(np.arange(0,n2), t[val], 1/resolution)
        f[:n2, 4] = -sin(np.arange(0,n2), t[val], 1/resolution)
        f[n2:2*n2+1, 2] = sin(np.arange(n2, 3*n2+1, 2), t[val], 1/resolution)
        f[n2:2*n2+1, 4] = -sin(np.arange(n2, 3*n2+1, 2), t[val], 1/resolution)
        f[2*n2+1:, 2] = sin(np.arange(3*n2+1,4*n2), t[val], 1/resolution)
        f[2*n2+1:, 4] = -sin(np.arange(3*n2+1,4*n2), t[val], 1/resolution)
            
        l4.set_data(x, f[:,2])
        fig.canvas.draw_idle()


    stime.on_changed(updatePlot)
    py.show()
    return fig, ax

def plotImw(): 
    
    Ezdata1 = np.load("data/Ezdata_ConstGrid.npy")
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    
    
    diff_res1 =  39*4 + 1
    resolutions1 = np.linspace(1, 40, diff_res1)
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    w1 = np.zeros(157)
    w1alt = []
    w1Index = 0
    
    for j in range(157):
        resSelect = j
        resolution = resolutions1[resSelect]
        t_eval = np.linspace(0, 2*resolution+1, (int(np.round((2*resolution+1)*10)) + 1))
        Ez = Ezdata1[resSelect,:int(resolution*8),:t_eval.shape[0]]
        
        if (resolution * 2 % 1 == 0):
            xIdx = np.arange(0,Ez.shape[0],2)
            
            Ezn = Ez[xIdx,:]
            tIdx = np.arange(0, Ez.shape[1], 5)
            
            w1alt.append(np.zeros(Ezn.shape[0]))
            
            for k in range(Ezn.shape[0]):
                xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
                w1alt[w1Index][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
            
            w1Index += 1
    
        wtemp = np.polyfit(t_eval,np.log(np.abs(Ez).max(0)),1)
        w1[j] = wtemp[0]
    
    w2 = np.zeros(77)
    w2alt = []
    for j in range(77):
        resSelect = j
        resolution = resolutions2[resSelect]
        
        xIdx = getxIdx(resolution)
        
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
        
        Ezn = Ez[xIdx,:]
        tIdx = np.arange(0, Ez.shape[1], 5)
        
        w2alt.append(np.zeros(Ezn.shape[0]))
        
        for k in range(Ezn.shape[0]):
            xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
            w2alt[j][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
    
        wtemp = np.polyfit(t_eval,np.log(np.abs(Ez).max(0)),1)
        w2[j] = wtemp[0]
    
    
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'

        wAnalyt1 = disp1(2*np.pi / np.linspace(1,40,200), 1/4)
        wAnalyt2 = disp2(2*np.pi / np.linspace(2,40,200), 1/4)
        py.scatter(resolutions1,-w1,s=2, label="constant grid, fine")
        py.scatter(resolutions2,-w2,s=2, label="two resolutions grid")
        py.plot(np.linspace(1,40,200),np.imag(wAnalyt1),label="constant grid, fine")
        py.plot(np.linspace(2,40,200),np.imag(wAnalyt2),label="constant grid, coarse")
        py.yscale("log")
        py.xlabel("\(n \; \, (k=2\pi/n)\)")
        py.ylabel("\(Im(\omega)\)")
        py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)

def plotImwNonConst():     
    #Ezdata1 = np.load("data/Ezdata_ConstGrid.npy")
    Ezdata1 = np.load("data/Ezdatao6.npy")
    
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    
    diff_res1 =  38*2 + 1
    resolutions1 = np.linspace(2, 40, diff_res1)
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    w1 = np.zeros(77)
    w1alt = []
    w1Index = 0
    
    for j in range(len(w1)):
        resSelect = j
        resolution = resolutions1[resSelect]
        
        xIdx = getxIdx(resolution)
        
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata1[resSelect,:int(resolution*6),:t_eval.shape[0]]
                
        Ezn = Ez[xIdx,:]
        tIdx = np.arange(0, Ez.shape[1], 5)
        
        w1alt.append(np.zeros(Ezn.shape[0]))
        
        for k in range(Ezn.shape[0]):
            xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
            w1alt[w1Index][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
        
        w1Index += 1

        wtemp = np.polyfit(t_eval,np.log(np.abs(Ez).max(0)),1)
        w1[j] = wtemp[0]
    
    w2 = np.zeros(77)
    w2alt = []
    for j in range(77):
        resSelect = j
        resolution = resolutions2[resSelect]
        
        xIdx = getxIdx(resolution)
        
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
        
        Ezn = Ez[xIdx,:]
        tIdx = np.arange(0, Ez.shape[1], 5)
        
        w2alt.append(np.zeros(Ezn.shape[0]))
        
        for k in range(Ezn.shape[0]):
            xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
            w2alt[j][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
    
        wtemp = np.polyfit(t_eval,np.log(np.abs(Ez).max(0)),1)
        w2[j] = wtemp[0]
    
    
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'

        wAnalyt1 = disp1(2*np.pi / np.linspace(1,40,200), 1/4)
        wAnalyt2 = disp2(2*np.pi / np.linspace(2,40,200), 1/4)
        wAnalyto6 = dispo6((2*np.pi) / np.linspace(2,40,200), 1/4)
        py.plot(np.linspace(1,40,200),np.imag(wAnalyt1),label="constant grid, order 4, fine")
        py.plot(np.linspace(2,40,200),np.imag(wAnalyto6),label="constant grid. order 6, fine")
        py.scatter(resolutions1, -w1, s=7, label="two resolutions grid, order 6", color="red")
        py.scatter(resolutions2, -w2, s=7, label="two resolutions grid, order 4")
        py.yscale("log")
        py.xlabel("\(n \; \, (k=2\pi/n)\)")
        py.ylabel("\(Im(\omega)\)")
        py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)


def plotRew():
    
    Ezdata1 = np.load("data/Ezdata_ConstGrid.npy")
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    
    
    diff_res1 =  39*4 + 1
    resolutions1 = np.linspace(1, 40, diff_res1)
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    w1 = np.zeros(157)
    w1alt = []
    w1Index = 0
    
    for j in range(157):
        resSelect = j
        resolution = resolutions1[resSelect]
        t_eval = np.linspace(0, 2*resolution+1, (int(np.round((2*resolution+1)*10)) + 1))
        Ez = Ezdata1[resSelect,:int(resolution*8),:t_eval.shape[0]]
        
        if (resolution * 2 % 1 == 0):
            xIdx = np.arange(0,Ez.shape[0],2)
            
            Ezn = Ez[xIdx,:]
            tIdx = np.arange(0, Ez.shape[1], 5)
            
            w1alt.append(np.zeros(Ezn.shape[0]))
            
            for k in range(Ezn.shape[0]):
                xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
                w1alt[w1Index][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
            
            w1Index += 1
    
    w2 = np.zeros(77)
    w2alt = []
    for j in range(77):
        resSelect = j
        resolution = resolutions2[resSelect]
        
        xIdx = getxIdx(resolution)
        
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
        
        Ezn = Ez[xIdx,:]
        tIdx = np.arange(0, Ez.shape[1], 5)
        
        w2alt.append(np.zeros(Ezn.shape[0]))
        
        for k in range(Ezn.shape[0]):
            xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
            w2alt[j][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
        
    
    wAnalyt1 = disp1(2*np.pi / np.linspace(1,40,200), 1/4)
    wAnalyt2 = disp2(2*np.pi / np.linspace(2,40,200), 1/4)

    ReWError1 = np.asarray([w1alt[i][0] for i in range(len(w1alt))])
    ReWError2 = np.asarray([w2alt[i][0] for i in range(len(w2alt))])
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'
        py.plot(np.linspace(1,40,200),np.real(wAnalyt1) - 2*np.pi / np.linspace(1,40,200),label="constant grid, fine")
        py.plot(np.linspace(2,40,200),np.real(wAnalyt2) - 2*np.pi / np.linspace(2,40,200),label="constant grid, coarse")
        py.scatter(resolutions1[np.where(resolutions1*2 % 1 == 0)],-ReWError1,s=2, label="constant grid, fine")
        py.scatter(resolutions2, ReWError2, s=2, label="two resolutions grid")
        py.yscale("log")
        py.xlabel("\(n \; \, (k=2\pi/n)\)")
        py.ylabel("\(Re(\omega) - k\)")
        py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)

def plotRewNonConst():
    Ezdata1 = np.load("data/Ezdatao6.npy")
    
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    
    
    diff_res1 =  38*2 + 1
    resolutions1 = np.linspace(2, 40, diff_res1)
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    w1 = np.zeros(77)
    w1alt = []
    w1Index = 0
    
    for j in range(77):
        resSelect = j
        resolution = resolutions1[resSelect]
        
        xIdx = getxIdx(resolution)
        
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata1[resSelect,:int(resolution*6),:t_eval.shape[0]]
        
        Ezn = Ez[xIdx,:]
        tIdx = np.arange(0, Ez.shape[1], 5)
                
        w1alt.append(np.zeros(Ezn.shape[0]))
        
        for k in range(Ezn.shape[0]):
            xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
            w1alt[j][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
        

    
    w2 = np.zeros(77)
    w2alt = []
    for j in range(77):
        resSelect = j
        resolution = resolutions2[resSelect]
        
        xIdx = getxIdx(resolution)
        
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
        
        Ezn = Ez[xIdx,:]
        tIdx = np.arange(0, Ez.shape[1], 5)
        
        w2alt.append(np.zeros(Ezn.shape[0]))
        
        for k in range(Ezn.shape[0]):
            xIdxTemp = np.arange(k, tIdx.shape[0] + k) % int(resolution * 4)
            w2alt[j][k] = np.polyfit(xIdxTemp/2,Ezn[xIdxTemp,tIdx],1)[0]
        
    
    wAnalyt1 = dispo6(2*np.pi / np.linspace(2,40,200), 1/4)
    wAnalyt2 = disp1(2*np.pi / np.linspace(2,40,200), 1/4)

    ReWError1 = np.asarray([w1alt[i][0] for i in range(len(w1alt))])
    ReWError2 = np.asarray([w2alt[i][0] for i in range(len(w2alt))])
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'
        
        py.plot(np.linspace(2,40,200),np.real(wAnalyt2) - 2*np.pi / np.linspace(2,40,200),label="constant grid, order 4, fine")
        py.plot(np.linspace(2,40,200),np.real(wAnalyt1) - 2*np.pi / np.linspace(2,40,200),label="constant grid, order 6, fine")
        
        py.scatter(resolutions2,ReWError2,s=7, label="two resolutions grid, order 4")
        py.scatter(resolutions1,ReWError1,s=7, color='red', label="two resolutions grid, order 6")
        
        py.yscale("log")
        py.xlabel("\(n  \; \, (k=2\pi/n)\)")
        py.ylabel("\(Re(\omega) - k\)")
        py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)

def plotSumofFields():
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    Hydata2 = np.load("data/Hydata.npy")
    
    s = np.sqrt(2)
    Rx = np.asarray([[s,  0,  0,  0,  0,  0],
                       [0,  1,  0,  0,  0, -1],
                       [0,  0,  1,  0,  1,  0],
                       [0,  0,  0,  s,  0,  0],
                       [0,  1,  0,  0,  0,  1],
                       [0,  0,  1,  0,  -1,  0]]) / s
    
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    for j in range(60,61):
        resSelect = j
        resolution = resolutions2[resSelect]
        
             
        t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
        Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
        Hy = Hydata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
        f = np.zeros((Ez.shape[0],Ez.shape[1],6))
        f[:, :, 2] = Ez
        f[:, :, 4] = Hy
        ftrans = np.tensordot(f,Rx, axes=([2,1]))
        with py.style.context("bmh"):
            py.figure()
            matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
            py.rcParams['axes.facecolor'] = 'none'
            py.plot(t_eval, np.abs(ftrans[:,:,2]).sum(0), label="\( E_z + B_y (i = 3)\)")
            py.plot(t_eval, np.abs(ftrans[:,:,5]).sum(0), label="\( E_z - B_y (i = 6)\)")
            py.yscale("log")
            py.xlabel("\(t\)")
            py.ylabel(r"\(\sum_j \vec{\bm{f}}^i_j(t)\)")
            py.legend(facecolor='white')
            py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)
 
def plotEzGridPoint():
    Ezdata2 = np.load("data/Ezdatao6.npy") #_ConstGrid.npy")
    Hydata2 = np.load("data/Hydatao6.npy")
        
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    resSelect = 76
    
    resolution = resolutions2[resSelect]
    t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
    Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
    Hy = Hydata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
    
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'
    
        py.plot([0,Ez.shape[0]],[ 1,  1], color='grey', alpha = 0.8)
        py.plot([0,Ez.shape[0]],[-1, -1], color='grey', alpha = 0.8)
        
        py.plot(Ez[:,0],label="\(E_z (i = 3)\)")
        py.plot(Hy[:,0],label="\(H_y (i = 5)\)")
    
        
        py.xlabel("grid position \(j\)")
        py.ylabel("\(f^i_j(t=0)\)")
        py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)

def plotEzPhysical():
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    Hydata2 = np.load("data/Hydata.npy")
        
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    resSelect = 76
    
    resolution = resolutions2[resSelect]
    n2 = resolution*2
    x = np.concatenate([np.arange(0,n2), np.arange(n2, 3*n2+1, 2), np.arange(3*n2+1,4*n2)], axis=0) / 4
    
    t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
    Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
    Hy = Hydata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
    
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'
    
        py.plot([0,x[-1]],[ 1,  1], color='grey', alpha = 0.8)
        py.plot([0,x[-1]],[-1, -1], color='grey', alpha = 0.8)
        
        py.plot(x, Ez[:,1],label="\(E_z\)")
        py.plot(x, Hy[:,1],label="\(H_y\)")
    
        py.xlabel("\(x\)")
        py.ylabel("\(f(x,t=0)\)")
        py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)


def errorLines():
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    resSelect = 76
    resolution = resolutions2[resSelect]
    xIdx = getxIdx(resolution)
    t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
    Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
    Ezn = Ez[xIdx,:]
    tIdx = np.arange(0, Ez.shape[1], 5)
    
    with py.style.context("bmh"):
        py.figure()
        matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
        py.rcParams['axes.facecolor'] = 'none'
    
        for i in range(Ezn.shape[0]):
            py.plot(tIdx,Ezn[np.arange(i, tIdx.shape[0] + i) % int(resolution*4), tIdx] - Ezn[np.arange(i, tIdx.shape[0] + i) % int(resolution*4), tIdx].mean(), linewidth=1.5)
    
        py.xlabel("\(t\)")
        py.ylabel(r"\(E_z - \bar{E_z}\)")
        #py.legend(facecolor="white")
        py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)

def plotDispersionRelation():
    kn = np.linspace(-4*np.pi, 0, 200)
    kp = np.linspace(0, 4*np.pi, 200)
    wk  = disp1(kp,1/4)
    wkn = disp1negative(kn,1/4)
    
    for i in range(2):
        with py.style.context("bmh"):
            py.figure()
            matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
            py.rcParams['axes.facecolor'] = 'none'
            
            if i == 0:
                py.plot(kp, np.real(wk),  label=r"\(Re\left(\omega^{+}\right)\)")
                py.plot(kp, np.imag(wk),  label=r"\(Im\left(\omega^{+}\right)\)")
                py.plot([kp[0], kp[-1]],[kp[0], kp[-1]], label=r"\(w_{\text{vacuum}}\)")
                py.xticks(np.pi*np.asarray([0,1,2,3,4]),[r"\(0\)", r"\(\frac{1}{4}\pi\)", r"\(\frac{2}{4}\pi\)", r"\(\frac{3}{4}\pi\)", r"\(\pi\)"])
               
            if i == 1:            
                py.plot(kn, np.real(wkn),  label=r"\(Re\left(\omega^{-}\right)\)")
                py.plot(kn, np.imag(wkn),  label=r"\(Im\left(\omega^{-}\right)\)")
                py.plot([kn[0],kn[-1]],[-kn[0], kn[-1]], label=r"\(\omega_{\text{vacuum}}\)")
                py.xticks(np.pi*np.asarray([-4,-3,-2,-1,0]),[r"\(-\pi\)",r"\(-\frac{3}{4}\pi\)",r"\(-\frac{2}{4}\pi\)",r"\(-\frac{1}{4}\pi\)",r"\(0\)"])
            
            if i == 2:
                k= np.linspace(-4*np.pi, 4*np.pi, 200)

                wk  = disp1(kp,1/4)
                wkn = disp1negative(kn,1/4)
                w = np.concatenate([wkn, wk],axis =0)
                py.plot(k, np.real(w),  label=r"\(Re\left(\omega\right)\)")
                py.plot(k, np.imag(w),  label=r"\(Im\left(\omega\right)\)")
                py.plot(k,np.abs(k), label=r"\(w_{\text{vacuum}}\)")
                py.xticks(np.pi*np.asarray([-4,-3,-2,-1,0,1,2,3,4]),[r"\(-\pi\)",r"\(-\frac{3}{4}\pi\)",r"\(-\frac{2}{4}\pi\)",r"\(-\frac{1}{4}\pi\)",r"\(0\)", r"\(\frac{1}{4}\pi\)", r"\(\frac{2}{4}\pi\)", r"\(\frac{3}{4}\pi\)", r"\(\pi\)"])
                
        
            py.xlabel("\(k \cdot \Delta\)")
            py.ylabel("\(w(k)\)")
            py.legend(facecolor="white")
            py.subplots_adjust(top=0.986,bottom=0.104,left=0.139,right=0.98)

def plotDispersionRelationResChange():
    def point1Negative(kx, delta):
        a = - 1
        b = +20 * np.exp(2j * kx * delta)
        c = -80 * np.exp(3j * kx * delta) 
        d = +45 * np.exp(4j * kx * delta)  
        e = +16 * np.exp(5j * kx * delta)
        return 1j*np.exp(-4j*kx*delta)*(a + b + c + d + e)/(60 * delta)
    def point2Negative(kx, delta):
        a = -  3
        b = + 25 * np.exp(2j * kx * delta)
        c = -225 * np.exp(4j * kx * delta) 
        d = +128 * np.exp(5j * kx * delta)  
        e = + 75 * np.exp(6j * kx * delta)
        return 1j*np.exp(-5j*kx*delta)*(a + b + c + d + e)/(240 * delta)
    def point3Negative(kx, delta):
        a = - 10
        b = + 63 * np.exp(2j * kx * delta)
        c = -210 * np.exp(4j * kx * delta) 
        d = - 35 * np.exp(6j * kx * delta)  
        e = +192 * np.exp(7j * kx * delta)
        return 1j*np.exp(-6j*kx*delta)*(a + b + c + d + e)/(420 * delta)
    def point3Positive(kx, delta):
        a = + 80
        b = -120 * np.exp( 1j * kx * delta)
        c = +  3 * np.exp(-2j * kx * delta) 
        d = + 45 * np.exp( 2j * kx * delta)  
        e = -  8 * np.exp( 3j * kx * delta)
        return 1j*(a + b + c + d + e)/(60 * delta) 
    def point4Positive(kx, delta):
        a = + 35
        b = +  6 * np.exp(-2j * kx * delta)
        c = - 90 * np.exp( 2j * kx * delta) 
        d = + 64 * np.exp( 3j * kx * delta)  
        e = - 15 * np.exp( 4j * kx * delta)
        return 1j*(a + b + c + d + e)/(60 * delta)
    def point5Positive(kx, delta):
        a = +189
        b = + 50 * np.exp(-2j * kx * delta)
        c = -350 * np.exp( 2j * kx * delta) 
        d = +175 * np.exp( 4j * kx * delta)  
        e = - 64 * np.exp( 5j * kx * delta)
        return 1j*(a + b + c + d + e)/(420 * delta)
    
    kn = np.linspace(-2*np.pi, 0, 200)
    kp = np.linspace(0, 2*np.pi, 200)
    
    p0n = disp1negative(kn,1/4)
    p1n = point1Negative(kn, 1/4)
    p2n = point2Negative(kn, 1/4)
    p3n = point3Negative(kn, 1/4)
    pin = disp1negative(kn, 1/2)
    
    p0p = disp1(kp,1/4)
    p3p = point3Positive(kp, 1/4)
    p4p = point4Positive(kp, 1/4)
    p5p = point5Positive(kp, 1/4)
    pip = disp2(kp,1/4)
    
    for i in range(4):
        with py.style.context("bmh"):
            py.figure()
            matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
            py.rcParams['axes.facecolor'] = 'none'
            
            if i == 0:
                py.plot([kn[0], kn[-1]],[-kn[0], kn[-1]], color='grey', label=r"\(w_{\text{vacuum}}\)")
                py.plot(kn, np.real(p0n), label=r"\(Re\left(\omega^{-}_{-\infty}\right)\)")
                py.plot(kn, np.real(p1n), label=r"\(Re\left(\omega^{-}_1\right)\)")
                py.plot(kn, np.real(p2n), label=r"\(Re\left(\omega^{-}_2\right)\)")
                py.plot(kn, np.real(p3n), label=r"\(Re\left(\omega^{-}_3\right)\)")
                py.plot(kn, np.real(pin), label=r"\(Re\left(\omega^{-}_{\infty}\right)\)")
                #py.xticks(np.pi*np.asarray([-4,-3,-2,-1,0]),[r"\(-4\pi\)",r"\(-3\pi\)",r"\(-2\pi\)",r"\(-1\pi\)",r"\(0\)"])
                py.xticks(np.pi*np.asarray([-2,-3/2,-1,-1/2,0]),[r"\(-\frac{4}{8}\pi\)",r"\(-\frac{3}{8}\pi\)",r"\(-\frac{2}{8}\pi\)",r"\(-\frac{1}{8}\pi\)",r"\(0\)"])
                py.legend(facecolor="white")
                
            if i == 1: 
                py.plot(kn, np.imag(p0n), label=r"\(Im\left(\omega^{-}_{-\infty}\right)\)")
                py.plot(kn, np.imag(p1n), label=r"\(Im\left(\omega^{-}_1\right)\)")
                py.plot(kn, np.imag(p2n), label=r"\(Im\left(\omega^{-}_2\right)\)")
                py.plot(kn, np.imag(p3n), label=r"\(Im\left(\omega^{-}_3\right)\)")
                py.plot(kn, np.imag(pin), label=r"\(Im\left(\omega^{-}_{\infty}\right)\)")
                #py.plot([kn[0], kn[-1]],[-kn[0], kn[-1]], label=r"\(w_{\text{vacuum}}\)")
                #py.xticks(np.pi*np.asarray([-4,-3,-2,-1,0]),[r"\(-4\pi\)",r"\(-3\pi\)",r"\(-2\pi\)",r"\(-1\pi\)",r"\(0\)"])
                py.xticks(np.pi*np.asarray([-2,-3/2,-1,-1/2,0]),[r"\(-\frac{4}{8}\pi\)",r"\(-\frac{3}{8}\pi\)",r"\(-\frac{2}{8}\pi\)",r"\(-\frac{1}{8}\pi\)",r"\(0\)"])
                py.legend(facecolor="white")
                
            if i == 2: 
                py.plot([kp[0],kp[-1]],[kp[0], kp[-1]], color='grey', label=r"\(\omega_{\text{vacuum}}\)")
                py.plot(kp, np.real(p0p),  label=r"\(Re\left(\omega^{+}_{-\infty}\right)\)")
                py.plot(kp, np.real(p3p),  label=r"\(Re\left(\omega^{+}_3\right)\)")
                py.plot(kp, np.real(p4p),  label=r"\(Re\left(\omega^{+}_4\right)\)")
                py.plot(kp, np.real(p5p),  label=r"\(Re\left(\omega^{+}_5\right)\)")
                py.plot(kp, np.real(pip),  label=r"\(Re\left(\omega^{+}_{\infty}\right)\)")
                
                #py.xticks(np.pi*np.asarray([0,1,2,3,4]),[r"\(0\)", r"\(1\pi\)", r"\(2\pi\)", r"\(3\pi\)", r"\(4\pi\)"])
                py.xticks(np.pi*np.asarray([0,1/2,1,3/2,2]),[r"\(0\)", r"\(\frac{1}{8}\pi\)", r"\(\frac{2}{8}\pi\)", r"\(\frac{3}{8}\pi\)", r"\(\frac{4}{8}\pi\)"])
                py.legend(facecolor="white")
            
            if i == 3:
                py.plot(kp, np.imag(p0p),  label=r"\(Im\left(\omega^{+}_{-\infty}\right)\)")
                py.plot(kp, np.imag(p3p),  label=r"\(Im\left(\omega^{+}_3\right)\)")
                py.plot(kp, np.imag(p4p),  label=r"\(Im\left(\omega^{+}_4\right)\)")
                py.plot(kp, np.imag(p5p),  label=r"\(Im\left(\omega^{+}_5\right)\)")
                py.plot(kp, np.imag(pip),  label=r"\(Im\left(\omega^{+}_{\infty}\right)\)")
                #py.plot([kp[0],kp[-1]],[kp[0], kp[-1]], label=r"\(\omega_{\text{vacuum}}\)")
                #py.xticks(np.pi*np.asarray([0,1,2,3,4]),[r"\(0\)", r"\(1\pi\)", r"\(2\pi\)", r"\(3\pi\)", r"\(4\pi\)"])
                py.xticks(np.pi*np.asarray([0,1/2,1,3/2,2]),[r"\(0\)", r"\(\frac{1}{8}\pi\)", r"\(\frac{2}{8}\pi\)", r"\(\frac{3}{8}\pi\)", r"\(\frac{4}{8}\pi\)"])
                py.legend(facecolor="white")
            
            
            py.xlabel("\(k\cdot\Delta\)")
            py.ylabel("\(w(k)\)")
            #py.legend(facecolor="white")
            py.subplots_adjust(top=0.986,bottom=0.104,left=0.139)
                               
def plotDispersionRelationResChangeInterpolate():
    def point3Negative(kx, delta):
        a = +2
        b = +6 * np.exp( 1j * kx * delta)
        c = -7 * np.exp(-2j * kx * delta) 
        d = -1 * np.exp(-4j * kx * delta)  
        return 1j*(a + b + c + d)/(24 * delta)
    def point3Positive(kx, delta):
        a = +23
        b = -36 * np.exp( 1j * kx * delta)
        c = + 3 * np.exp(-2j * kx * delta) 
        d = +12 * np.exp( 2j * kx * delta)  
        e = - 2 * np.exp( 3j * kx * delta)
        return 1j*(a + b + c + d + e)/(24 * delta)
    
    kn = np.linspace(-2*np.pi, 0, 200)
    kp = np.linspace(0, 2*np.pi, 200)
    
    p0n = disp1negative(kn,1/4)
    p0p = disp1(kp,1/4)
    
    pin = disp1negative(kn, 1/2)
    pip = disp2(kp,1/4)
    
    p3n = point3Negative(kn,1/4)
    p3p = point3Positive(kp,1/4)

    for i in range(2):
         with py.style.context("bmh"):
             py.figure()
             matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
             py.rcParams['axes.facecolor'] = 'none'
             
             if i == 0:
                 py.plot([kn[0], kn[-1]],[-kn[0], kn[-1]], color='grey', label=r"\(w_{\text{vacuum}}\)")
                 py.plot(kn, np.real(p3n), label=r"\(Re\left(\omega^{-}_3\right)\)")
                 py.plot(kn, np.imag(p3n), label=r"\(Im\left(\omega^{-}_3\right)\)")
                 
                 #py.xticks(np.pi*np.asarray([-4,-3,-2,-1,0]),[r"\(-4\pi\)",r"\(-3\pi\)",r"\(-2\pi\)",r"\(-1\pi\)",r"\(0\)"])
                 py.xticks(np.pi*np.asarray([-2,-3/2,-1,-1/2,0]),[r"\(-\frac{4}{8}\pi\)",r"\(-\frac{3}{8}\pi\)",r"\(-\frac{2}{8}\pi\)",r"\(-\frac{1}{8}\pi\)",r"\(0\)"])
                 py.legend(facecolor="white")
                 
             if i == 1: 
                 py.plot([kp[0],kp[-1]],[kp[0], kp[-1]], color='grey', label=r"\(\omega_{\text{vacuum}}\)")
                 py.plot(kp, np.real(p3p),  label=r"\(Re\left(\omega^{+}_3\right)\)")
                 py.plot(kp, np.imag(p3p),  label=r"\(Im\left(\omega^{+}_3\right)\)")
                 #py.xticks(np.pi*np.asarray([0,1,2,3,4]),[r"\(0\)", r"\(1\pi\)", r"\(2\pi\)", r"\(3\pi\)", r"\(4\pi\)"])
                 py.xticks(np.pi*np.asarray([0,1/2,1,3/2,2]),[r"\(0\)", r"\(\frac{1}{8}\pi\)", r"\(\frac{2}{8}\pi\)", r"\(\frac{3}{8}\pi\)", r"\(\frac{4}{8}\pi\)"])
                 py.legend(facecolor="white")
             py.xlabel("\(k\cdot\Delta\)")
             py.ylabel("\(w(k)\)")
             py.subplots_adjust(top=0.986,bottom=0.104,left=0.139)

def errorLinesCompare():
    Ezdata1 = np.load("data/Ezdata_ConstGrid.npy") 
    Ezdata2 = np.load("data/Ezdata.npy") #_ConstGrid.npy")

    diff_res1 =  39*4 + 1
    resolutions1 = np.linspace(1, 40, diff_res1)
    
    diff_res2 =  38*2 + 1
    resolutions2 = np.linspace(2, 40, diff_res2)
    
    resSelect = 156
    
    resolution = resolutions1[resSelect]
    t_eval1 = np.linspace(0, 2*resolution+1, (int(np.round((2*resolution+1)*10)) + 1))
    Ez1 = Ezdata1[resSelect,:int(resolution*8),:t_eval1.shape[0]]
    

    xIdx1 = np.arange(0,Ez1.shape[0],2)
        
    Ezn1 = Ez1[xIdx1,:]
    tIdx1 = np.arange(0, Ez1.shape[1], 5)

    
    resSelect = 76
    resolution = resolutions2[resSelect]
    xIdx = getxIdx(resolution)
    t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
    Ez = Ezdata2[resSelect,:int(resolution*6),:t_eval.shape[0]]
    Ezn = Ez[xIdx,:]
    tIdx = np.arange(0, Ez.shape[1], 5)
    for j in range(2):
        with py.style.context("bmh"):
            py.figure()
            matplotlib.rc('axes',edgecolor='black',linewidth=1.1)
            py.rcParams['axes.facecolor'] = 'none'
        
            #for i in range(Ezn.shape[0]):
            
            if j == 0:
                i = 0
            if j == 1:  
                i = 60
                py.yticks([1, 1-5e-7, 1-1e-6, 1-1.5e-6, 1-2e-6], [0, -0.5, -1, -1.5, -2]) #1E-6
                py.title(r"\(\times 10^{-6} + 1\)", fontdict={'fontsize': 14}, loc="left", y=0.987)
                
            fit = np.polyfit(tIdx1, Ezn1[np.arange(i+40, tIdx1.shape[0] + i+40) % int(resolution*2), tIdx1], 1)
            py.plot(tIdx,Ezn[np.arange(i, tIdx.shape[0] + i) % int(resolution*4), tIdx], label=r"\(\Delta_1,\Delta_2\)")
            #py.plot(tIdx1,Ezn1[np.arange(i+40, tIdx1.shape[0] + i+40) % int(resolution*2), tIdx1], linewidth=1.5)
            py.plot(tIdx, fit[1] + fit[0]*tIdx, label=r"\(\Delta_1\)")
            
            
    
            py.xlabel("\(t\)")
            py.ylabel(r"\(E_z\) ")
            py.legend(facecolor="white")
            py.subplots_adjust(top=0.946,bottom=0.104,left=0.139,right=0.98)
            
        
#plotImw()
plotRew()

#plotImwNonConst()
plotRewNonConst()
# Ezdata = np.load("data/Ezdatao6.npy")
# Ez = Ezdata[-1]
# plotEzAdjust(Ez, 40)

# funcs = [plotRew, plotImw, errorLines]

# for fun in funcs:
#     fun()
#errorLines()
#errorLinesCompare()
#plotDispersionRelation()
#plotDispersionRelationResChange()
#plotDispersionRelationResChangeInterpolate()