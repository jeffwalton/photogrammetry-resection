import numpy as np
from scipy.optimize import minimize


def collinearityEqnResidual(iop,eop,x,y,X,Y,Z):
    from math import sin, cos
    x0 = iop['x0']
    y0 = iop['y0']
    focallength = iop['f']

    om = eop['omega']
    ph = eop['phi']
    kp = eop['kappa']

    XL = eop['XL']
    YL = eop['YL']
    ZL = eop['ZL']

    Mom = np.matrix([[1, 0, 0], [0, cos(om), sin(om)], [0, -sin(om), cos(om)]])
    Mph = np.matrix([[cos(ph), 0, -sin(ph)], [0, 1, 0], [sin(ph), 0, cos(ph)]])
    Mkp = np.matrix([[cos(kp), sin(kp), 0], [-sin(kp), cos(kp), 0], [0, 0, 1]])

    M = Mkp * Mph * Mom

    uvw = M * np.matrix([[X-XL], [Y-YL], [Z-ZL]])
    #print uvw

    #res = np.zeros((2,1))
    #res[0,0] = x - x0 + focallength * uvw[0] / uvw[2]
    #res[1,0] = y - y0 + focallength * uvw[1] / uvw[2]
    resx = x - x0 + focallength * uvw[0,0] / uvw[2,0]
    resy = y - y0 + focallength * uvw[1,0] / uvw[2,0]

    return resx, resy




class collinearityData:
    def __init__(self):

        f = open('cam.inp','r')
        dat = np.loadtxt(f,float)
        f.close

        self.eop = {}

        self.eop['omega'] = dat[0]
        self.eop['phi'] = dat[1]
        self.eop['kappa'] = dat[2]

        self.eop['XL'] = dat[3]
        self.eop['YL'] = dat[4]
        self.eop['ZL'] = dat[5]

        self.iop = {}

        self.iop['x0'] = dat[6]
        self.iop['y0'] = dat[7]
        self.iop['f'] = dat[8]

        self.label = []
        x = []
        y = []
        X = []
        Y = []
        Z = []

        f = open('resect.inp','r')
        for line in f:
            l = line.split()
            self.label.append(l[0])
            x.append(float(l[1]))
            y.append(float(l[2]))
            X.append(float(l[3]))
            Y.append(float(l[4]))
            Z.append(float(l[5]))

        self.x = np.array(x)
        self.y = np.array(y)
        self.X = np.array(X)
        self.Y = np.array(Y)
        self.Z = np.array(Z)



def coll_func(indepVars):
    global data
    iop = data.iop
    #eop = data.eop
    label = data.label
    x = data.x
    y = data.y
    X = data.X
    Y = data.Y
    Z = data.Z

    eop = {}
    eop['omega'] = indepVars[0]
    eop['phi'] = indepVars[1]
    eop['kappa'] = indepVars[2]
    eop['XL'] = indepVars[3]
    eop['YL'] = indepVars[4]
    eop['ZL'] = indepVars[5]

    i = 0
    F = 0.0
    for l in label:

        F1, F2 = collinearityEqnResidual(iop,eop,x[i],y[i],X[i],Y[i],Z[i])
        F += F1**2 + F2**2
        i += 1

    return F


data = collinearityData()
x0 = np.zeros(6)
eop = data.eop
#print eop
x0[0] = eop['omega']
x0[1] = eop['phi']
x0[2] = eop['kappa']
x0[3] = eop['XL']
x0[4] = eop['YL']
x0[5] = eop['ZL']
#print x0
#print data.iop
#print data.x
#print data.y
#print data.X
#print data.Y
#print data.Z

#res = minimize(coll_func, x0, method='BFGS', options={'disp': True})
res = minimize(coll_func, x0, options={'disp': True})
print res.x
