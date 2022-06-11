import numpy as np
import matplotlib.pyplot as py
from matplotlib.widgets import Slider
from matplotlib import animation, rc

from scipy.integrate import solve_ivp

dim_x = 500
dim_y = 1
dim_z = 1
bound = 3

def sin(x, t):
    w = 2 * np.pi * 1/50
    return np.sin(w/6 * x - w * t)

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

    sMinus = np.asarray([-1/12, 1/2, -3/2, 5/6, 1/4, 0, 0]) * 6
    sPlus  = np.asarray([0, 0, -1/4, -5/6, 3/2, -1/2, 1/12]) * 6
    # sMinus = np.asarray([-1, 1, 0]) * 1
    # sPlus  = np.asarray([0, -1, 1]) * 1



    return sPlus, sMinus

def derivative_f(y, R, sPlus, sMinus):

    def getSvloc(sp, sm, num):
        return np.diag([sp[num], sp[num], sp[num], sm[num], sm[num], sm[num]])

    df = np.zeros_like(y)
    sLen = len(sPlus)//2
    sPs1 = np.asarray([0, 0, -50/180, -126/180, 225/180, -50/180, 1/180])*6           # -1, 0, 1, 2, 4
    sPs2 = np.asarray([0, 0, -441/1260, -495/1260, 980/1260, -49/1260, 5/1260])*6    # -1, 0, 1, 3, 5
    sPs3 = 3*np.asarray([0, 0, -729/1260, 490/1260, 315/1260, -90/1260, 14/1260])*6          # -0.5, 0, 1, 2, 3

    sMs1 = np.asarray([-10/20, 54/60, -135/20, 90/20, 1/20, 0, 0])*6             # -1.5, -1, -0.5, 0, 1
    sMs2 = np.asarray([-189/140, 675/140, -700/140, 189/140, 25/140, 0, 0])*6             # -2, -1.5, -1, 0, 1
    sMs3 = 3*np.asarray([-243/1260, 490/1260, -735/210, 390/1260, 98/1260, 0, 0])*6    # -2.5, -2, -1, 0, 1

    for j in range(3, dim_x + bound): #dim_x + 2*bound
        for i in range(len(sPlus)):
            Sv = np.diag([sPlus[i], sPlus[i], sPlus[i], sMinus[i], sMinus[i], sMinus[i]])
            if j == 401:
                Sv = getSvloc(sPs1, sMinus, i)
            if j == 402:
                Sv = getSvloc(sPs2, sMinus, i)
            if j == 403:
                Sv = getSvloc(sPs3, sMs1, i)
            if j == 404:
                Sv = getSvloc(sPlus, sMs2, i)
            if j == 405:
                Sv = getSvloc(sPlus, sMs3, i)

            if j >= 403:# and j <= 250:
               Sv = Sv / 3

            temp = y[0, 0, (j - sLen + i) % (dim_x + 2 * bound), :]
            temp = np.matmul(R[0], temp)
            temp = np.matmul(Sv, temp)
            temp = np.matmul(R[0].T, temp)
            df[0, 0, j, :] += temp

    return df


def dfdt(t, y, B, R, sPlus, sMinus):
    y = y.reshape(dim_z, dim_y, dim_x + 2 * bound, 6)
    #for i in range(3):
    x = np.arange(-2, 1, 1)
    #y[0, 0, 198:201, 2] = gauss(x, 2 * t)#sin(x, t)
    #y[0, 0, 198:201, 4] = gauss(x, 2 * t)#sin(x, t)

    # if(t<100):
    #     y[0, 0, 0:3, 2] = sin(x, t)  #gauss(-x, t)
    #     y[0, 0, 0:3, 4] = -sin(x, t)  #gauss(-x, t)
    # else:
    #     y[0, 0, 0:3, 2] = 0
    #     y[0, 0, 0:3, 4] = 0

    #y[0, 0, 100, 5] = 1.0/np.sqrt(2) * gauss(t)

    df = derivative_f(y, R, sPlus, sMinus)
    dtf = np.zeros_like(df)
    for j in range(0, dim_x + 2*bound):
        if j>=3: #(j < 198 or j > 201):
            dtf[0, 0, j, :] = np.matmul(B[0], df[0, 0, j, :])

    # dtf[0, 0, 3:, 2] = dtf[0, 0, 3:, 2]/9
    #dtf[0, 0, :, 4] = dtf[0, 0, :, 4]

    return dtf.flatten()


B = get_B()
R = get_R()

sPlus, sMinus = get_Stencil()
f = np.zeros((dim_z, dim_y, dim_x + 2 * bound, 6))
f[0, 0, bound:304, 2] = sin(np.arange(0,301), 0)
f[0, 0, bound:304, 4] = -sin(np.arange(0,301), 0)
# f[0, 0, 204:304, 2] = sin(np.arange(202,401, 2), 0)/3
# f[0, 0, 204:304, 4] = -sin(np.arange(202, 401, 2), 0)


f = f.flatten()
t_eval = np.linspace(0, 261, 261*4 + 1)
out = solve_ivp(dfdt, (0, 261), f.flatten(), args=(B, R, sPlus, sMinus), rtol=1e-6, atol=1e-6, t_eval=t_eval)


print(out)
y = out.y
t = out.t
y= y.reshape((dim_x + 2 * bound, 6, -1))

Ez = y[:, 2, :]
Hy = y[:, 4, :]
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
a = [np.sum(np.abs(Ez[:203, i])) + np.sum(np.abs(Ez[204:, i]))*2 + np.sum((0.75*Ez[203, i] + 0.25*Ez[204, i]))*1.5 for i in range(Ez.shape[1])]
b = [np.sum(np.abs(Ez[1:204:2, i]))*2 + np.sum(np.abs(Ez[204:, i]))*2 for i in range(Ez.shape[1])]
