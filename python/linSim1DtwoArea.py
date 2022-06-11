import numpy as np
import matplotlib.pyplot as py
from matplotlib.widgets import Slider
from matplotlib import animation, rc
import matplotlib 

from scipy.integrate import solve_ivp

matplotlib.rcParams["figure.dpi"] = 200

dim_x = 1200
dim_y = 1
dim_z = 1
bound = 0

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

    sMinus = np.asarray([-1/12, 1/2, -3/2, 5/6, 1/4, 0, 0]) * 4
    sPlus  = np.asarray([0, 0, -1/4, -5/6, 3/2, -1/2, 1/12]) * 4
    # sMinus = np.asarray([-1, 1, 0]) * 1
    # sPlus  = np.asarray([0, -1, 1]) * 1



    return sPlus, sMinus

def derivative_f(y, R, sPlus, sMinus):

    def getSvloc(sp, sm, num):
        return np.diag([sp[num], sp[num], sp[num], sm[num], sm[num], sm[num]])

    df = np.zeros_like(y)
    sLen = len(sPlus)//2
    sPs1 = np.asarray([0, 0, -16/60, -45/60, 80/60, -20/60, 1/60])*4            # -1, 0, 1, 2, 4
    sPs2 = np.asarray([0, 0, -75/240, -128/240, 225/240, - 25/240, 3/240])*4    # -1, 0, 1, 3, 5
    sPs3 = np.asarray([0, 0, -192/210, +35/210, 1, -63/210, 10/210])*4          # -0.5, 0, 1, 2, 3

    sMs1 = np.asarray([-8/30, 45/30, -120/30, 80/30, 3/30, 0, 0])*4             # -1.5, -1, -0.5, 0, 1
    sMs2 = np.asarray([-15/30, 64/30, -90/30, 35/30, 6/30, 0, 0])*4             # -2, -1.5, -1, 0, 1
    sMs3 = np.asarray([-64/210, 175/210, -350/210, 189/210, 50/210, 0, 0])*4    # -2.5, -2, -1, 0, 1
    
    sPs1b = np.asarray([0, 0, -50/210, -189/210, 350/210, -175/210, 64/210])*4 		# -1, 0, 1, 2, 2.5
    sPs2b = np.asarray([0, 0, -6/30, -35/30, 90/30, -64/30, 15/30])*4                 # -1, 0, 1, 1.5, 2
    sPs3b = np.asarray([0, 0, -3/60, -80/60, 120/60, -45/60, 8/60 ])*4 	          	# -1, 0, 0.5, 1, 1.5

    sMs1b = np.asarray([-10/420, 63/420, -210/420, -35/420, 192/420, 0, 0])*4       	#-6, -4, -2, 0, 1
    sMs2b = np.asarray([-3/240, 25/240, -225/240, 128/240, 75/240, 0, 0])*4           	#-5, -3, -1, 0, 1
    sMs3b = np.asarray([-1/60, 20/60, -80/60, 45/60, 16/60, 0, 0])*4 	                	#-4, -2, -1, 0, 1

    for j in range(0, dim_x + 2*bound): #dim_x + 2*bound
        for i in range(len(sPlus)):


            Sv = np.diag([sPlus[i], sPlus[i], sPlus[i], sMinus[i], sMinus[i], sMinus[i]])
            if j == 398:
                Sv = getSvloc(sPs1, sMinus, i)
            if j == 399:
                Sv = getSvloc(sPs2, sMinus, i)
            if j == 400:
                Sv = getSvloc(sPs3, sMs1, i)
            if j == 401:
                Sv = getSvloc(sPlus, sMs2, i)
            if j == 402:
                Sv = getSvloc(sPlus, sMs3, i)
            
            
                
            if j == 798:
                Sv = getSvloc(sPs1b, sMinus, i)
            if j == 799:
                Sv = getSvloc(sPs2b, sMinus, i)
            if j == 800:
                Sv = getSvloc(sPs3b, sMs1b, i)
            if j == 801:
                Sv = getSvloc(sPlus, sMs2b, i)
            if j == 802:
                Sv = getSvloc(sPlus, sMs3b, i)
            
            if j >= 400 and j < 800:# and j <= 250:
                Sv = Sv / 2
            
            # if j == 198:
            #     Sv = getSvloc(sPs1, sMinus, i)
            # if j == 199:
            #     Sv = getSvloc(sPs2, sMinus, i)
            # if j == 200:
            #     Sv = getSvloc(sPs3, sMs1, i)
            # if j == 201:
            #     Sv = getSvloc(sPlus, sMs2, i)
            # if j == 202:
            #     Sv = getSvloc(sPlus, sMs3, i)
            
            
                
            # if j == 398:
            #     Sv = getSvloc(sPs1b, sMinus, i)
            # if j == 399:
            #     Sv = getSvloc(sPs2b, sMinus, i)
            # if j == 400:
            #     Sv = getSvloc(sPs3b, sMs1b, i)
            # if j == 401:
            #     Sv = getSvloc(sPlus, sMs2b, i)
            # if j == 402:
            #     Sv = getSvloc(sPlus, sMs3b, i)
                
            # if j >= 200 and j < 400:# and j <= 250:
            #     Sv = Sv / 2

            temp = y[0, 0, (j - sLen + i) % (dim_x + 2 * bound), :]
            temp = np.matmul(R[0], temp)
            temp = np.matmul(Sv, temp)
            
            df[0, 0, j, :] += temp
        df[0, 0, j, :] =  np.matmul(R[0].T, df[0, 0, j, :])

    return df


def dfdt(t, y, B, R, sPlus, sMinus):
    y = y.reshape(dim_z, dim_y, dim_x + 2 * bound, 6)

    df = derivative_f(y, R, sPlus, sMinus)
    dtf = np.zeros_like(df)
    
    for j in range(0, dim_x + 2*bound):
        #if j>=3: #(j < 198 or j > 201):
        dtf[0, 0, j, :] = np.matmul(B[0], df[0, 0, j, :])


    return dtf.flatten()


res = np.zeros((200,9))
opt = np.zeros(200)
B = get_B()
R = get_R()

sPlus, sMinus = get_Stencil()

#resolutions = np.linspace(1, 12, 200)
#Ezs = np.zeros((200, dim_x + 2*bound, (int(np.round(resolutions[-1]))+1)*10 + 1))


# resolution = 50 #resolutions[i]
# f = np.zeros((dim_z, dim_y, dim_x + 2 * bound, 6))
# f[0, 0, :200, 2] = sin(np.arange(0,200), 0, 1/resolution)
# f[0, 0, :200, 4] = -sin(np.arange(0,200), 0, 1/resolution)
# f[0, 0, 200:401, 2] = sin(np.arange(200, 601, 2), 0, 1/resolution)
# f[0, 0, 200:401, 4] = -sin(np.arange(200, 601, 2), 0, 1/resolution)
# f[0, 0, 401:, 2] = sin(np.arange(601,800), 0, 1/resolution)
# f[0, 0, 401:, 4] = -sin(np.arange(601,800), 0, 1/resolution)

resolution = 200 
f = np.zeros((dim_z, dim_y, dim_x + 2 * bound, 6))
f[0, 0, :400, 2] = sin(np.arange(0,400), 0, 1/resolution)
f[0, 0, :400, 4] = -sin(np.arange(0,400), 0, 1/resolution)
f[0, 0, 400:801, 2] = sin(np.arange(400, 1201, 2), 0, 1/resolution)
f[0, 0, 400:801, 4] = -sin(np.arange(400, 1201, 2), 0, 1/resolution)
f[0, 0, 801:, 2] = sin(np.arange(1201,1600), 0, 1/resolution)
f[0, 0, 801:, 4] = -sin(np.arange(1201,1600), 0, 1/resolution)

#resolution = 200 #resolutions[i]
# f = np.zeros((dim_z, dim_y, dim_x + 2 * bound, 6))
# f[0, 0, :, 2] = sin(np.arange(0,dim_x), 0, 1/resolution)
# f[0, 0, :, 4] = -sin(np.arange(0,dim_x), 0, 1/resolution)


f = f.flatten()
t_eval = np.linspace(0, 5*resolution, (int(np.round((5*resolution)*10)) + 1))
out = solve_ivp(dfdt, (0, 1000), f.flatten(), t_eval=t_eval, args=(B, R, sPlus, sMinus), rtol=1e-12, atol=1e-12)

y = out.y
t = out.t
y= y.reshape((dim_x + 2 * bound, 6, -1))

Ez = y[:, 2, :]
Hy = y[:, 4, :]

def error(xg, x, tg, t):   
    err = Ez[xg,tg] - sin(x[xg],t[tg],1/200)
    return err

x = np.concatenate([np.arange(0,400),np.arange(400, 1201, 2),np.arange(1201,1600)])
xi = np.arange(0,Ez.shape[0])
ti = np.arange(0,Ez.shape[1])
xg, tg = np.meshgrid(xi,ti)


e = error(xg,x,tg,t)

fig = py.figure()
ax = py.axes(projection="3d")
ax.plot_surface(xg,tg,e,cmap='Greens',edgecolor='grey',alpha=0.9)

x = np.concatenate([np.arange(0,400),np.arange(400, 1201, 2),np.arange(1201,1600)])
ti = np.arange(0,Ez.shape[1])
xg, tg = np.meshgrid(x,ti)

py.contour(xg,tg,e)


#np.save('Ezdata.npy', Ezs)
#np.save('res.npy',res)
#np.save('opt.npy',opt)
#print(out)
'''
Rxf = np.tensordot(y, R[0], axes=([1], [1]))  # f_ijk R_lj = Rxf_ikl
S_forward = Rxf[:, :, 5]**2
S_backward = Rxf[:, :, 2]**2


fig, ax = py.subplots()
py.subplots_adjust(left=0.25, bottom=0.20)

x = np.arange(Ez.shape[0])

l1, = ax.plot(x, Ez[:, 0], color='green')
l2, = ax.plot(x, Hy[:, 0], color='blue', alpha=0.5)
#l4, = ax.plot(np.arange(Ez.shape[0]), gauss(np.arange(Ez.shape[0])- 200, 0), color="red", alpha=0.3)

l3, = ax.plot([0, x[-1]], [1.0, 1.0])
l3, = ax.plot([0, x[-1]], [-1.0, -1.0])

#ax.plot([x[150], x[250]], [1, 1], 'ro')
#py.ylim([-1, 1])
ax.margins(x=0)

axcolor = 'white'
axtime = py.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

stime = Slider(axtime, 'Timestep', 0, Ez.shape[1] - 1, valinit=0, valstep=1)

def updatePlot(val):

    ax.set_title('t = ' + str(t[val]))
    l1.set_data(x, Ez[:, val])
    l2.set_data(x, Hy[:, val])
    #l4.set_data(x, gauss(np.arange(Ez.shape[0])- 200, t[val]))
    #l4.set_data(x, correct)

    fig.canvas.draw_idle()
    #return (l, )

#anim = animation.FuncAnimation(fig, updatePlot, frames=Ez.shape[1] - 1, interval=20, blit=True)
#anim.save('./animation.gif', fps=60)
stime.on_changed(updatePlot)
py.show()


a = np.zeros(Ez.shape[1])
for i in range(Ez.shape[1]):
    a[i] = np.sum(Ez[2:123, i]**2) + np.sum((Ez[128:-4, i]**2))*2 + 1.5*Ez[123, i]**2 + 1.5*Ez[124, i]**2 + 1.5*Ez[125, i]**2 + 1.5*Ez[126, i]**2 + 1.5*Ez[127, i]**2
py.plot(t, a)
py.show()
a = [np.sum(Ez[:203, i]**2) + np.sum(Ez[204:, i]**2)*2 + 1.5*Ez[203, i]**2 for i in range(Ez.shape[1])]
b = [np.sum(Ez[1:204:2, i]**2)*2 + np.sum(Ez[204:, i]**2)*2 for i in range(Ez.shape[1])]
#
'''