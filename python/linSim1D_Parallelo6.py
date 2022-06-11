import numpy as np
import matplotlib.pyplot as py
from matplotlib.widgets import Slider
from matplotlib import animation, rc
import matplotlib 
import os 


from scipy.integrate import solve_ivp
from multiprocessing import Pool, Array

matplotlib.rcParams["figure.dpi"] = 200

#dim_x = 400
dim_y = 1
dim_z = 1
bound = 0

sPs1 	= np.asarray([0, 0, 15, -175, -224, 525, -175, 35, -1])*4/420  					# -2,-1,0,1,2,3,5
sPs2 	= np.asarray([0, 0, 35, -384, -350, 896, -210, 14, -1])*4/840 					# -2,-1,0,1,2,4,6
sPs3 	= np.asarray([0, 0, 1120, -11025, -3552, 14700, -1470, 252, -25])*4/20160 		# -2,-1,0,1,3,5,7
sPs4 	= np.asarray([0, 0, 252, -2048, 1155, 840, -252, 60, -7])*4/2520 				# -2,-1,0,2,4,6,8
sPs5 	= np.asarray([0, 0, 2048, -8316, -5775, 16632, -5940, 1540, -189])*4/27720 		# -3,-2,0,2,4,6,8

sMs1 	= np.asarray([9, -70, 252, -630, 315, 126, -2, 0, 0])*4/420  					# -4,-3,-2,-1,0,1,3
sMs2	= np.asarray([35, -256, 840, -1792, 1120, 56, -3, 0, 0])*4/840  					# -4,-3,-2,-1,0,2,4
sMs3	= np.asarray([256, -1575, 3840, -4200, 1344, 360, -25, 0, 0])*4/2520  			# -5,-4,-3,-2,0,2,4
sMs4	= np.asarray([210, -1024, 1575, -2100, 924, 450, -35, 0, 0])*4/2520  			# -6,-5,-4,-2,0,2,4
sMs5 	= np.asarray([1024, -3234, 8085, -19404, 8580, 5390, -441, 0, 0])*4/27720  		# -7,-6,-4,-2,0,2,4


sPs1b 	= np.asarray([0, 0, 441, -5390, -8580, 19404, -8085, 3234, -1024])*4/27720  	# -4, -2, 0, 2, 4, 6, 7
sPs2b 	= np.asarray([0, 0, 35, -450, -924, 2100, -1575, 1024, -210])*4/2520 			# -4, -2, 0, 2, 4, 5, 6
sPs3b 	= np.asarray([0, 0, 25, -360, -1344, 4200, -3840, 1575, -256])*4/2520 			# -4, -2, 0, 2, 3, 4, 5
sPs4b 	= np.asarray([0, 0, 3, -56, -1120, 1792, -840, 256, -35])*4/840 				# -4, -2, 0, 1, 2, 3, 4
sPs5b 	= np.asarray([0, 0, 2, -126, -315, 630, -252, 70, -9])*4/420 					# -3, -1, 0, 1, 2, 3, 4

sMs1b 	= np.asarray([189, -1540, 5940, -16632, 5775, 8316, -2048, 0, 0])*4/27720  		# -8, -6, -4, -2, 0, 2, 3
sMs2b	= np.asarray([7, -60, 252, -840, -1155, 2048, -252, 0, 0])*4/2520  				# -8, -6, -4, -2, 0, 1, 2
sMs3b	= np.asarray([25, -252, 1470, -14700, 3552, 11025, -1120, 0, 0])*4/20160  		# -7, -5, -3, -1, 0, 1, 2
sMs4b	= np.asarray([1, -14, 210, -896, 350, 384, -35, 0, 0])*4/840				  	# -6, -4, -2, -1, 0, 1, 2
sMs5b	= np.asarray([1, -35, 175, -525, 224, 175, -15, 0, 0])*4/420			 		# -5, -3, -2, -1, 0, 1, 2




def sin(x, t, w = 1/50):
    w *= 2 * np.pi
    return np.sin(w/4 * x - w * t)

def gauss(x, t):
    return np.exp(-(x + (t-60))**2/400)


def get_B():
    Q = np.array([object, object, object])
    Q[0] = np.asarray([[ 0,  0,  0],
                      [ 0,  0, -1],
                      [ 0,  1,  0]])

    Q[1] = np.asarray([[ 0,  0,  1],
                      [ 0,  0,  0],
                      [-1,  0,  0]])

    Q[2] = np.asarray([[ 0, -1,  0],
                      [ 1,  0,  0],
                      [ 0,  0,  0]])

    zero3 = np.asarray(np.zeros((3, 3)))
    B = np.asarray([object, object, object])

    for i in range(0, 3):
        B[i] = np.block([[zero3, Q[i]], [-Q[i], zero3]])

    return B

def get_R():
    R = np.array([object, object, object])
    s = np.sqrt(2)
    R[0] = np.asarray([[s,  0,  0,  0,  0,  0],
                       [0,  1,  0,  0,  0, -1],
                       [0,  0,  1,  0,  1,  0],
                       [0,  0,  0,  s,  0,  0],
                       [0,  1,  0,  0,  0,  1],
                       [0,  0,  1,  0,  -1,  0]]) / s

    R[1] = np.asarray([[ 0,  s,  0,  0,  0,  0],
                       [-1,  1,  0,  0,  0, -1],
                       [ 0,  0, -1,  1,  0,  0],
                       [ 0,  0,  0,  0,  s,  0],
                       [-1,  1,  0,  0,  0,  1],
                       [ 0,  0, -1, -1,  0,  0]]) / s

    R[2] = np.asarray([[0,  0,  s,  0,  0,  0],
                       [1,  0,  0,  0, -1,  0],
                       [0,  1,  0,  1,  0,  0],
                       [0,  0,  0,  0,  0,  s],
                       [1,  0,  0,  0,  1,  0],
                       [0,  1,  0, -1,  0,  0]]) / s
    return R

def get_Stencil():
    # sPlus  = np.asarray([-1/12, 1/2, -3/2, 5/6, 1/4, 0, 0]) * 2
    # sMinus = np.asarray([0, 0, -1/4, -5/6, 3/2, -1/2, 1/12]) * 2
    # sPlus  = np.asarray([-1, 1, 0]) * 3
    # sMinus = np.asarray([0, -1, 1]) * 3


    #sPlus  = np.asarray([0, 1/20, -1/2, -1/3, 1.0, -1/4, 1/30]) * 3
    #sMinus = np.asarray([-1/30, 1/4, -1.0, 1/3, 1/2, -1/20, 0]) * 3

    sMinus = np.asarray([1, -8, 30, -80, 35, 24, -2, 0, 0]) * 4 * 1/60
    sPlus  = np.asarray([0, 0, 2, -24, -35, 80, -30, 8, -1]) * 4 * 1/60
    # sMinus = np.asarray([-1, 1, 0]) * 1
    # sPlus  = np.asarray([0, -1, 1]) * 1

    return sPlus, sMinus

def derivative_f(y, R, sPlus, sMinus, dim_x):

    def getSvloc(sp, sm, num):
        return np.diag([sp[num], sp[num], sp[num], sm[num], sm[num], sm[num]])

    df = np.zeros_like(y)
    sLen = len(sPlus)//2

    n2 = dim_x//3

    for j in range(0, dim_x + bound): #dim_x + 2*bound
        for i in range(len(sPlus)):


            Sv = np.diag([sPlus[i], sPlus[i], sPlus[i], sMinus[i], sMinus[i], sMinus[i]])

            if j == (n2 - 3):
                Sv = getSvloc(sPs1, sMinus, i)
            if j == (n2 - 2):
                Sv = getSvloc(sPs2, sMinus, i)
            if j == (n2 - 1):
                Sv = getSvloc(sPs3, sMs1, i)
            if j == (n2):
                # adding a factor 2 because the stencils are all derived in the 
                # fine resolution
                Sv = getSvloc(2*sPs4, 2*sMs2, i)
            if j == (n2 + 1):
                Sv = getSvloc(2*sPs5, 2*sMs3, i)
            if j == (n2 + 2):
                Sv = getSvloc(sPlus, 2*sMs4, i)
            if j == (n2 + 3):
                Sv = getSvloc(sPlus, 2*sMs5, i)
            
            
            if j == (2*n2 - 3):
                Sv = getSvloc(2*sPs1b, sMinus, i)    
            if j == (2*n2 - 2):
                Sv = getSvloc(2*sPs2b, sMinus, i)
            if j == (2*n2 - 1):
                Sv = getSvloc(2*sPs3b, 2*sMs1b, i)
            if j == (2*n2):
                Sv = getSvloc(sPs4b, sMs2b, i)
            if j == (2*n2 + 1):
                Sv = getSvloc(sPs5b, sMs3b, i)
            if j == (2*n2 + 2):
                Sv = getSvloc(sPlus, sMs4b, i)
            if j == (2*n2 + 3):
                Sv = getSvloc(sPlus, sMs5b, i)
            
            if j >= n2 and j < 2*n2:# and j <= 250:
                Sv = Sv / 2            

            temp = y[0, 0, (j - sLen + i) % (dim_x + 2 * bound), :]
            temp = np.matmul(R[0], temp)
            temp = np.matmul(Sv, temp)

            df[0, 0, j, :] += temp
            
        df[0, 0, j, :] =  np.matmul(R[0].T, df[0, 0, j, :])
    return df


def dfdt(t, y, B, R, sPlus, sMinus, dim_x):
    y = y.reshape(dim_z, dim_y, dim_x + 2 * bound, 6)
    #for i in range(3):
    #x = np.arange(-2, 1, 1)
    #y[0, 0, 198:201, 2] = gauss(x, 2 * t)#sin(x, t)
    #y[0, 0, 198:201, 4] = gauss(x, 2 * t)#sin(x, t)

    # if(t<100):
    #     y[0, 0, 0:3, 2] = sin(x, t)  #gauss(-x, t)
    #     y[0, 0, 0:3, 4] = -sin(x, t)  #gauss(-x, t)
    # else:
    #     y[0, 0, 0:3, 2] = 0
    #     y[0, 0, 0:3, 4] = 0

    #y[0, 0, 100, 5] = 1.0/np.sqrt(2) * gauss(t)

    df = derivative_f(y, R, sPlus, sMinus, dim_x)
    dtf = np.zeros_like(df)
    for j in range(0, dim_x + bound):
        dtf[0, 0, j, :] = np.matmul(B[0], df[0, 0, j, :])

    #dtf[0, 0, 3:, 2] = dtf[0, 0, 3:, 2]/9
    #dtf[0, 0, :, 4] = dtf[0, 0, :, 4]

    return dtf.flatten()


def runSim(i):
    Ezs = np.ctypeslib.as_array(Ezs_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
    Hys = np.ctypeslib.as_array(Hys_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
    resolution = resolutions[i]
    
    simDim_x = int(resolution * 3 * 2) # two full oscillations fit onto the grid (3->avg resolution, 2->#oscillations)
    n2 = simDim_x//3
        
    f = np.zeros((dim_z, dim_y, simDim_x + 2 * bound, 6))
    f[0, 0, :n2, 2] = sin(np.arange(0,n2), 0, 1/resolution)
    f[0, 0, :n2, 4] = -sin(np.arange(0,n2), 0, 1/resolution)
    f[0, 0, n2:2*n2+1, 2] = sin(np.arange(n2, 3*n2+1, 2), 0, 1/resolution)
    f[0, 0, n2:2*n2+1, 4] = -sin(np.arange(n2, 3*n2+1, 2), 0, 1/resolution)
    f[0, 0, 2*n2+1:, 2] = sin(np.arange(3*n2+1,4*n2), 0, 1/resolution)
    f[0, 0, 2*n2+1:, 4] = -sin(np.arange(3*n2+1,4*n2), 0, 1/resolution)
    
    
    f = f.flatten()
    t_eval = np.linspace(0, 6*resolution+1, (int(np.round((6*resolution+1)*10)) + 1))
    out = solve_ivp(dfdt, (0, 6*resolution+1), f.flatten(), t_eval = t_eval,  args=(B, R, sPlus, sMinus, simDim_x), rtol=1e-12, atol=1e-12)
    
    y = out.y
    y= y.reshape((simDim_x + 2 * bound, 6, -1))
    
    Ez = y[:, 2, :]
    Hy = y[:, 4, :]
    Ezs[i, :simDim_x, :int(np.round((6*resolution+1)*10)) + 1] = Ez       
    Hys[i, :simDim_x, :int(np.round((6*resolution+1)*10)) + 1] = Hy  
    print(str(i) + ' / ' + str(diff_res))

diff_res =  38*2 + 1
dim_x = 40 * 3 * 2 

resolutions = np.linspace(2, 40, diff_res)
maxtRes = (int(np.round(6*resolutions[-1]))+1)*10 + 1

B = get_B()
R = get_R()

sPlus, sMinus = get_Stencil()

Ezs_Ctypes = Array('d', diff_res * (dim_x + 2*bound) * maxtRes, lock=False)
Hys_Ctypes = Array('d', diff_res * (dim_x + 2*bound) * maxtRes, lock=False)

numb_of_CPU = os.getenv("SLURM_CPUS_PER_TASK")
if numb_of_CPU is None:
    numb_of_CPU = 8
else:
    numb_of_CPU = int(numb_of_CPU)

with Pool(processes=numb_of_CPU) as pool:
    pool.map(runSim, range(diff_res))

Ezs = np.ctypeslib.as_array(Ezs_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
Hys = np.ctypeslib.as_array(Hys_Ctypes).reshape((diff_res, (dim_x + 2*bound), maxtRes)) 
np.save('Ezdata.npy', Ezs)
np.save('Hydata.npy', Hys)
