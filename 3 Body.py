import numpy as np
import matplotlib.pyplot as plt

a = 1
e = 0
P = 2 * np.pi
G = 1
delta = 0.1
a2 = a + delta
del_t = (1/500)*P
tmax = 1000*P
eta = 0.01
t = np.arange(0, tmax, del_t)

n = 3 #Here we are defining the number of bodies in the system whch we can change that. Also, we have to make sure that the size of the masses_all array should correspond to n

masses_all = np.array([1 - (2*1e-5), 1e-5, 1e-5])
masses = masses_all[:n]
M = np.sum(masses)

positions_all = np.array([
    [0, 0],  # Central body
    [a * (1 + e), 0],  # First body
    [-a2 * (1 + e), 0]  # Second body
])
positions = positions_all[:n]

velocities_all = np.array([
    [0, 0],  # Central body at rest
    [0, np.sqrt((G * M * (1 - e)) / (a * (1 + e)))],  # First body velocity
    [0, -np.sqrt((G * M * (1 - e)) / (a2 * (1 + e)))]  # Second body velocity
])
velocities = velocities_all[:n]

#CoM positions/velocities
com_pos = np.sum(masses[:, None] * positions, axis=0) / M
com_vel = np.sum(masses[:, None] * velocities, axis=0) / M
positions -= com_pos
velocities -= com_vel
del_t_init = del_t

def adaptive_time_step(acc, jerk, del_t_init):
    del_t = del_t_init
    for j in range(n):    
        n_dt = np.linalg.norm(acc[j])/np.linalg.norm(jerk[j])
        if n_dt < 1:
            del_t = del_t*n_dt
    return del_t
