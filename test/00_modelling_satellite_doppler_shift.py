# This module generates a satellite orbit, 
# then models the Doppler shift from the
# relative velocity to ground bases.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from astropy import units as u
from astropy import time
from astropy.coordinates import representation

from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.plotting.static import StaticOrbitPlotter

import emcee
import corner
import timeit


R_earth = 6378.1    # km
v_c     = 300000    # km/s

N_t     = 10        # Tota number of times that satellite signal is measured by each GS
deltaT  = 20        # seconds. Time interval between measurements


# Real orbital elements for International Space Station
# Data from ISS at 2019-apr-10

a    = (R_earth + 409) * u.km
ecc  =   0.0001586 * u.one
inc  =  51.6418    * u.deg
raan = 332.5814    * u.deg
argp = 179.7333    * u.deg
nu   =-179.6181    * u.deg

# Arbitrary Value, (few hours of diference to true data.)
EPOCH = time.Time('2019-04-10 12:00:00.0' , format='iso')


# Generate the orbit object "iss"
iss = Orbit.from_classical(Earth, a=a, ecc=ecc, inc=inc, raan=raan, argp=argp, nu=nu, epoch=EPOCH)



deltaT = deltaT * u.second

iss_propagation = []
iss_r = []
iss_v = []
iss_t = []


for i in range(20):
    aux = iss.propagate(deltaT * i)
    iss_propagation.append(aux)
    
    iss_r.append(aux.r )
    iss_v.append(aux.v )
    iss_t.append(aux.epoch.unix)
    

### Ground Base positions (in cartesian coordinates)

a = np.array([6206.87748242, -2681.198044  ,   604.2233833])
a /= np.linalg.norm(a)
a = a * R_earth


b = np.array([6250.0, -2700.0, 500.0])
b /= np.linalg.norm(b)
b = b * R_earth


c = np.array([6100.0, -2800.0, 500.0])
c /= np.linalg.norm(c)
c = c * R_earth

d = np.array([5800.0, -2500.0, 200.0])
d /= np.linalg.norm(d)
d = d * R_earth


BaseA = a
BaseB = b
BaseC = c
BaseD = d

iss_r = np.array(iss_r)
iss_v = np.array(iss_v)



### Plot satellite trajectory and ground bases

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.scatter(BaseA[0], BaseA[1], BaseA[2], label="GBA")
ax.scatter(BaseB[0], BaseB[1], BaseB[2], label="GBB")
ax.scatter(BaseC[0], BaseC[1], BaseC[2], label="GBC")
ax.scatter(BaseD[0], BaseD[1], BaseD[2], label="GBD")
ax.scatter(iss_r[:,0], iss_r[:,1], iss_r[:,2], s=10, label="satellite")

ax.legend(loc=0)
plt.show()




def distance(a, b):
    x0 = a[0]
    x1 = a[1]
    x2 = a[2]
    
    y0 = b[0]
    y1 = b[1]
    y2 = b[2]
    
    dist = (x0-y0)**2 + (x1-y1)**2 + (x2-y2)**2
    dist = dist**0.5
    return dist


def rel_vel_to_base(r, v, Base):
        """
        Returns the relative speed of the satellite to
        some ground location. This velocity is 
        positive if the satellite is going towards
        the ground location, negative if is going 
        outwards.
        """
        
        x1 = Base[0]
        y1 = Base[1]
        z1 = Base[2]
    
        x_r = r[0] - x1
        y_r = r[1] - y1
        z_r = r[2] - z1  

        d = ( x_r ** 2 + y_r ** 2 + z_r ** 2 )**0.5
    
        ux  = x_r / d
        uy  = y_r / d
        uz  = z_r / d
    
        u = np.array([ux, uy, uz])
        
        return np.inner(-u,v)

    
Real_Data = []
A=[]
B=[]
C=[]
D=[]
T = []

for t in range(len(iss_t)):
    A.append(rel_vel_to_base( iss_r[t], iss_v[t], BaseA) )
    B.append(rel_vel_to_base( iss_r[t], iss_v[t], BaseB) )
    C.append(rel_vel_to_base( iss_r[t], iss_v[t], BaseC) )
    D.append(rel_vel_to_base( iss_r[t], iss_v[t], BaseD) )
    T.append(t*deltaT.value)
    
A = np.array(A)
B = np.array(B)
C = np.array(C)
D = np.array(D)
T = np.array(T)    



### Plot Doppler Shift measured by Ground Bases.

fig = plt.figure()

plt.title("Satellite Relative Speed to Ground Bases")

plt.plot(T, A, label="GB A")
plt.plot(T, B, label="GB B")
plt.plot(T, C, label="GB C")
plt.plot(T, D, label="GB D")

plt.xlabel("Time (seconds)")
plt.ylabel("Velocity (m/s)")
plt.legend(loc=0)

plt.show()


### Store emulated satellite doppler shift data

MD_path = "measured_data/"

aa = np.column_stack((T,A))
MD_filename = "A.dat"
np.savetxt(MD_path + MD_filename, aa)

bb = np.column_stack((T,A))
MD_filename = "B.dat"
np.savetxt(MD_path + MD_filename, bb)

cc = np.column_stack((T,C))
MD_filename = "C.dat"
np.savetxt(MD_path + MD_filename, cc)

dd = np.column_stack((T,D))
MD_filename = "D.dat"
np.savetxt(MD_path + MD_filename, dd)
