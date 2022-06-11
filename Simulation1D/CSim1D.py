import ctypes
import numpy as np
import matplotlib.pyplot as py
from matplotlib.widgets import Slider

class Simulation_boundaryVtbl(ctypes.Structure):
        _fields_ = []

class Simulation(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("timesteps", ctypes.c_int),
        ("currtime", ctypes.c_int),
        ("imp0", ctypes.c_double),
        ("bVtblp", ctypes.POINTER(Simulation_boundaryVtbl)),
        ("Ezdata", ctypes.POINTER(ctypes.c_double)),
        ("Hydata", ctypes.POINTER(ctypes.c_double)),
    ]

libSim = ctypes.CDLL("../build/libSimWin1D.so")

Simulation_ctor = libSim.Simulation_ctor
Simulation_ctor.argtypes = [ctypes.POINTER(Simulation), ctypes.c_int, ctypes.c_size_t, ctypes.c_double, ctypes.c_double]
Simulation_ctor.restype = ctypes.c_void_p

# Simulation_set_E = libSim.Simulation_set_E
# Simulation_set_E.argtypes = [ctypes.POINTER(Simulation), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
# Simulation_set_E.restype = ctypes.c_void_p
#
# Simulation_set_B = libSim.Simulation_set_B
# Simulation_set_B.argtypes = [ctypes.POINTER(Simulation), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double]
# Simulation_set_B.restype = ctypes.c_void_p

Simulation_updateEz = libSim.Simulation_updateEz
Simulation_updateEz.argtypes = [ctypes.POINTER(Simulation)]
Simulation_updateEz.restype = ctypes.c_void_p

Simulation_updateHy = libSim.Simulation_updateHy
Simulation_updateHy.argtypes = [ctypes.POINTER(Simulation)]
Simulation_updateHy.restype = ctypes.c_void_p

Simulation_get_Ezdata = libSim.Simulation_get_Ezdata
Simulation_get_Ezdata.argtypes = [ctypes.POINTER(Simulation)]
Simulation_get_Ezdata.restype = ctypes.POINTER(ctypes.c_double)

Simulation_get_Hydata = libSim.Simulation_get_Ezdata
Simulation_get_Hydata.argtypes = [ctypes.POINTER(Simulation)]
Simulation_get_Hydata.restype = ctypes.POINTER(ctypes.c_double)

Simulation_run = libSim.Simulation_run
Simulation_run.argtypes = [ctypes.POINTER(Simulation)]
Simulation_run.restype = ctypes.c_void_p

Simulation_dtor = libSim.Simulation_dtor
Simulation_dtor.argtypes = [ctypes.POINTER(Simulation)]
Simulation_dtor.restype = ctypes.c_void_p

dim_x = 200
times = 1000

blub = Simulation()

timesteps   = ctypes.c_int(times)
size        = ctypes.c_size_t(dim_x)
deltaX      = ctypes.c_double(0.01)
deltaT      = ctypes.c_double(0.01)
imp0        = ctypes.c_double(377.0)
loss        = ctypes.c_double(0.0)


Simulation_ctor(ctypes.pointer(blub), timesteps, size, imp0, loss)
Simulation_run(ctypes.pointer(blub))

Ezdata = np.ctypeslib.as_array(Simulation_get_Ezdata(ctypes.pointer(blub)), (times, dim_x))
Hydata = np.ctypeslib.as_array(Simulation_get_Hydata(ctypes.pointer(blub)), (times, dim_x))

fig, ax = py.subplots()
py.subplots_adjust(left=0.25, bottom=0.20)
l1, = ax.plot(Ezdata[0, :], color='green')
l2, = ax.plot(Hydata[0, :], color='red', alpha=0.5)
py.ylim([-1, 1])
ax.margins(x=0)

axcolor = 'white'
axtime = py.axes([0.15, 0.1, 0.65, 0.03], facecolor=axcolor)

stime = Slider(axtime, 'Timestep', 0, times - 1, valinit=0, valstep=1)

def updatePlot(val):
    time = stime.val
    # print(time)
    # print(Ezdata[time, :])
    # print("")
    l1.set_ydata(Ezdata[time, :])
    l2.set_ydata(Hydata[time, :])
    #ax.relim()
    #ax.autoscale_view()
    #fig.canvas.draw_idle()

stime.on_changed(updatePlot)
py.show()


Simulation_dtor(ctypes.pointer(blub))
