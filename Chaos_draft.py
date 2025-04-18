import numpy as np
import matplotlib.pyplot as plt

a = 1
e = 0
P = 2 * np.pi
G = 1
delta = 0.05
a2 = a + delta
del_t = (1/500)*P
tmax = 100*P
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

def Euler(G, n, masses, positions, velocities, del_t_init, tmax):
    t = 0.0
    del_t = del_t_init
    i = 0

    time_array = []

    pos = [positions.copy()]
    vel = [velocities.copy()]
    E = []
    J = []
    e = []
    a = []
    Log_E = []
    Log_e = []
    Log_a = []
    at = [del_t]


    count = 0
    while t < tmax:
        acc = np.zeros((n, 2))
        jerk = np.zeros((n, 2))
        KE = 0
        PE = 0
        angular_momentum = 0

        for j in range(n):
            KE += 0.5 * masses[j] * np.dot(vel[i][j], vel[i][j])
            angular_momentum += masses[j] * (pos[i][j, 0] * vel[i][j, 1] - pos[i][j, 1] * vel[i][j, 0])

            for k in range(n):
                if j != k:
                    r_vec = pos[i][k] - pos[i][j]
                    r_mag = np.linalg.norm(r_vec)
                    v_vec = vel[i][j] - vel[i][k]

                    mu = G * (masses[j] + masses[k])
                    if r_mag != 0:
                        e_vec = (angular_momentum ** 2 / (mu * r_mag)) * r_vec - (r_vec / r_mag)
                        e_mag = np.linalg.norm(e_vec)

                        PE -= G * masses[j] * masses[k] / r_mag
                        acc[j] += G * masses[k] * r_vec / r_mag**3

                        if j < n - 1:
                            if len(e) <= i:
                                e.append(np.zeros(n - 1))
                                a.append(np.zeros(n - 1))
                                Log_e.append(np.zeros(n - 1))
                                Log_a.append(np.zeros(n - 1))
                            e[i][j] = e_mag
                            a[i][j] = (angular_momentum ** 2 / mu) / (1 - e_mag ** 2)
                            Log_e[i][j] = np.log10(abs((e_mag - e[0][j])/e[0][j]))
                            Log_a[i][j] = np.log10(abs((a[i][j] - a[0][j])/a[0][j]))

                        rvec = r_vec / r_mag
                        jerk[j] += G * masses[k] * (v_vec / r_mag**3 - 3 * np.dot(v_vec, rvec) * rvec / r_mag**5)

        # Adaptive time step
        # del_t = adaptive_time_step(acc, jerk, del_t_init)
        # at.append(del_t)
        
        E.append(KE + PE)
        J.append(angular_momentum)

        Log_E.append(np.log10(abs((E[i] - E[0])/E[0])))
        
        new_vel = vel[i] + acc * del_t
        new_pos = pos[i] + vel[i] * del_t

        vel.append(new_vel)
        pos.append(new_pos)

        t += del_t
        time_array.append(t)
        i += 1

    # Convert lists to arrays before returning
    return (
        np.array(pos),
        np.array(vel),
        np.array(E),
        np.array(J),
        np.array(e),
        np.array(a),
        np.array(at),
        np.array(time_array),
        np.array(Log_E),
        np.array(Log_e),
        np.array(Log_a)
    )


def Euler_Cromer(G, n, masses, positions, velocities, del_t_init, tmax):
    # Initialize simulation
    t = 0.0
    del_t = del_t_init
    time_array = []
    i = 1

    # Storage lists
    pos = [positions.copy()]
    vel = [velocities.copy()]
    E = []
    J = []
    e = []
    a = []
    Log_E = []
    Log_e = []
    Log_a = []
    
    at = [del_t]

    while t < tmax:
        acc = np.zeros((n, 2))
        jerk = np.zeros((n, 2))  # Required for adaptive step
        KE = 0
        PE = 0
        angular_momentum = 0

        for j in range(n):
            KE += 0.5 * masses[j] * np.dot(vel[i-1][j], vel[i-1][j])
            angular_momentum += masses[j] * (
                pos[i-1][j, 0] * vel[i-1][j, 1] - pos[i-1][j, 1] * vel[i-1][j, 0]
            )

        # Interactions
        for j in range(n):
            for k in range(j + 1, n):
                r_vec = pos[i-1][k] - pos[i-1][j]
                r_mag = np.linalg.norm(r_vec)
                v_vec = vel[i-1][j] - vel[i-1][k]

                if r_mag != 0:
                    mu = G * (masses[j] + masses[k])
                    e_vec = (angular_momentum**2 / (mu * r_mag)) * r_vec - (r_vec / r_mag)
                    e_mag = np.linalg.norm(e_vec)

                    PE -= G * masses[j] * masses[k] / r_mag
                    acc[j] += G * masses[k] * r_vec / r_mag**3
                    acc[k] -= G * masses[j] * r_vec / r_mag**3

                    # Fill e and a
                    if len(e) < i:
                        e.append(np.zeros(n - 1))
                        a.append(np.zeros(n - 1))
                        Log_e.append(np.zeros(n - 1))
                        Log_a.append(np.zeros(n - 1))
                    if j < n - 1:
                        e[i-1][j] = e_mag
                        a[i-1][j] = (angular_momentum**2 / mu) / (1 - e_mag**2)
                        Log_e[i-1][j-1] = np.log10(abs((e_mag - e[0][j])/e[0][j]))
                        Log_a[i-1][j-1] = np.log10(abs((a[i-1][j-1] - a[0][j-1])/a[0][j-1]))

                    rvec = r_vec / r_mag
                    jerk[j] += G * masses[k] * (v_vec / r_mag**3 - 3 * np.dot(v_vec, rvec) * rvec / r_mag**5)
                    jerk[k] -= G * masses[j] * (v_vec / r_mag**3 - 3 * np.dot(v_vec, rvec) * rvec / r_mag**5)

        # Time step update
        # del_t = adaptive_time_step(acc, jerk, del_t_init)
        # at.append(del_t)

        # Energy and angular momentum
        E.append(KE + PE)
        J.append(angular_momentum)

        Log_E.append(np.log10(abs((E[i-1] - E[0])/E[0])))

        # Velocity and position updates (Euler-Cromer: use updated velocity)
        new_vel = vel[i-1] + acc * del_t
        new_pos = pos[i-1] + new_vel * del_t

        vel.append(new_vel)
        pos.append(new_pos)

        t += del_t
        time_array.append(t)
        i += 1

    # Convert to NumPy arrays
    return (
        np.array(pos),
        np.array(vel),
        np.array(E),
        np.array(J),
        np.array(e),
        np.array(a),
        np.array(at),
        np.array(time_array),
        np.array(Log_E),
        np.array(Log_e),
        np.array(Log_a)
    )

def Leap_frog(G, n, masses, positions, velocities, del_t_init, tmax):
    t = 0.0
    time_array = []
    del_t = del_t_init
    i = 1

    pos = [positions.copy()]
    vel = [velocities.copy()]
    E = []
    J = []
    e = []
    a = []
    Log_E = []
    Log_e = []
    Log_a = []
    at = [del_t]

    while t < tmax:
        acc_half = np.zeros((n, 2))
        jerk = np.zeros((n, 2))  # Required for adaptive step

        pos_half = pos[i - 1] + vel[i - 1] * del_t / 2

        KE = 0
        PE = 0
        angular_momentum = 0

        for j in range(n):
            KE += 0.5 * masses[j] * np.dot(vel[i - 1][j], vel[i - 1][j])
            angular_momentum += masses[j] * (
                pos[i - 1][j, 0] * vel[i - 1][j, 1] - pos[i - 1][j, 1] * vel[i - 1][j, 0]
            )

            for k in range(n):
                if j != k:
                    r_vec = pos_half[k] - pos_half[j]
                    r_mag = np.linalg.norm(r_vec)

                    r_vec_full = pos[i - 1][k] - pos[i - 1][j]
                    r_mag_full = np.linalg.norm(r_vec_full)
                    v_vec = vel[i - 1][j] - vel[i - 1][k]

                    if r_mag_full != 0:
                        mu = G * (masses[j] + masses[k])
                        e_vec = (angular_momentum ** 2 / (mu * r_mag_full)) * r_vec - (r_vec_full / r_mag_full)
                        e_mag = np.linalg.norm(e_vec)

                        PE -= G * masses[j] * masses[k] / r_mag_full
                        acc_half[j] += G * masses[k] * r_vec / r_mag ** 3

                        if len(e) < i:
                            e.append(np.zeros(n - 1))
                            a.append(np.zeros(n - 1))
                            Log_e.append(np.zeros(n - 1))
                            Log_a.append(np.zeros(n - 1))
                        
                        if j < n - 1:
                            e[i - 1][j] = e_mag
                            a[i - 1][j] = (angular_momentum ** 2 / mu) / (1 - e_mag ** 2)
                            Log_e[i-1][j] = np.log10(abs((e_mag - e[0][j])/e[0][j]))
                            Log_a[i-1][j] = np.log10(abs((a[i-1][j] - a[0][j])/a[0][j]))

                        rvec = r_vec / r_mag
                        jerk[j] += G * masses[k] * (
                            v_vec / r_mag ** 3 - 3 * np.dot(v_vec, rvec) * rvec / r_mag ** 5
                        )

        # Adaptive step update
        # del_t = adaptive_time_step(acc_half, jerk, del_t_init)
        # at.append(del_t)

        E.append(KE + PE)
        J.append(angular_momentum)

        Log_E.append(np.log10(abs((E[i-1] - E[0])/E[0])))

        new_vel = vel[i - 1] + acc_half * del_t
        new_pos = pos_half + vel[i - 1] * del_t / 2

        vel.append(new_vel)
        pos.append(new_pos)

        t += del_t
        time_array.append(t)
        i += 1

    return (
        np.array(pos),
        np.array(vel),
        np.array(E),
        np.array(J),
        np.array(e),
        np.array(a),
        np.array(at),
        np.array(time_array),
        np.array(Log_E),
        np.array(Log_e),
        np.array(Log_a)
    )

def Verlet(G, n, masses, positions, velocities, del_t_init, tmax):
    t = 0.0
    time_array = [t]
    del_t = del_t_init
    i = 1
    
    pos = [positions.copy()]
    vel = [velocities.copy()]

    E = [0]  
    J = [0]  
    e = []  
    a = []  
    Log_E = []
    Log_e = []
    Log_a = []
    at = [del_t] 

    
    acc = np.zeros((n, 2))
    for j in range(n):
        for k in range(j + 1, n):
            r_vec = pos[0][k] - pos[0][j]
            r_mag = np.linalg.norm(r_vec)
            acc[j] += G * masses[k] * r_vec / r_mag**3
            acc[k] -= G * masses[j] * r_vec / r_mag**3

    while t < tmax:
        jerk = np.zeros((n, 2))  # Reset jerk for each time step
        KE = 0
        PE = 0
        J.append(0)  

        v_half = vel[i - 1] + acc * del_t / 2
        pos.append(pos[i - 1] + v_half * del_t)

        acc_new = np.zeros((n, 2))

        for j in range(n):
            KE += 0.5 * masses[j] * np.dot(vel[i - 1] [j], vel[i - 1] [j])
            J[i] += masses[j] * (pos[i - 1][j, 0] * vel[i - 1][j, 1] - pos[i - 1][j, 1] * vel[i - 1][j, 0])

            for k in range(j + 1, n):
                r_vec = pos[i][k] - pos[i][j]
                r_mag = np.linalg.norm(r_vec) + 1e-10 
                v_vec = vel[i - 1][j] - vel[i - 1][k]

                mu = G * (masses[j] + masses[k])  
                e_vec = (J[i - 1]**2 / (mu * r_mag)) * r_vec - (r_vec / r_mag)
                e_mag = np.linalg.norm(e_vec)

                PE -= G * masses[j] * masses[k] / r_mag
                acc_new[j] += G * masses[k] * r_vec / r_mag**3
                acc_new[k] -= G * masses[j] * r_vec / r_mag**3

                e.append(e_mag)
                a.append((J[i - 1]**2 / (mu)) / (1 - e_mag**2))
                epsilon = 1e-20
                Log_e.append(np.log10(abs((e_mag - e[0]) / e[0]) + epsilon))
                Log_a.append(np.log10(abs((a[-1] - a[0]) / a[0]) + epsilon))

                rvec = r_vec / r_mag
                jerk[j] += G * masses[k] * (v_vec / r_mag ** 3 - 3 * np.dot(v_vec, rvec) * rvec / r_mag ** 5)

        del_t = adaptive_time_step(acc_new, jerk, del_t_init)
        at.append(del_t)

        vel.append(v_half + acc_new * del_t / 2)
        acc = acc_new 

        E.append(KE + PE)
        epsilon = 1e-20 

        if abs(E[0]) > epsilon:
            rel_err_E = abs((E[i-1] - E[0]) / E[0])
        else:
            rel_err_E = abs(E[i-1])  # fallback if E[0] is too close to 0

        Log_E.append(np.log10(rel_err_E + epsilon)) 

        t += del_t
        time_array.append(t)
        i += 1

    return np.array(pos), np.array(vel), np.array(E), np.array(J), np.array(e), np.array(a), np.array(at), np.array(time_array), np.array(Log_E), np.array(Log_e), np.array(Log_a)

def RK4(positions, velocities, masses, G, del_t, n):

    def compute_accelerations(pos):
        acc = np.zeros_like(pos)
        for i in range(n):
            for j in range(n):
                if i != j:
                    r_vec = pos[j] - pos[i]
                    r_mag = np.linalg.norm(r_vec)
                    acc[i] += G * masses[j] * r_vec / r_mag**3
        return acc
    
    pos = np.zeros((len(t), n, 2))
    vel = np.zeros((len(t), n, 2))

    jerk = np.zeros((n, 2))

    pos[0] = positions
    vel[0] = velocities

    at = [del_t]

    E = np.zeros(len(t))
    J = np.zeros(len(t))

    e = np.zeros((len(t),n-1))
    a = np.zeros((len(t),n-1))
    
    Log_E = np.zeros(len(t))
    Log_e = np.zeros((len(t),n-1))
    Log_a = np.zeros((len(t),n-1))  

    acc = compute_accelerations(positions)

    del_t = adaptive_time_step(acc, jerk, del_t_init)
    at.append(del_t)
    

    for i in range(1, len(t)):
        
        k1_v = compute_accelerations(positions) * del_t
        k1_x = velocities * del_t

        k2_v = compute_accelerations(positions + k1_x / 2) * del_t
        k2_x = (velocities + k1_v / 2) * del_t

        k3_v = compute_accelerations(positions + k2_x / 2) * del_t
        k3_x = (velocities + k2_v / 2) * del_t

        k4_v = compute_accelerations(positions + k3_x) * del_t
        k4_x = (velocities + k3_v) * del_t

        positions += (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6
        velocities += (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

        pos[i] = positions
        vel[i] = velocities

        KE = 0
        PE = 0
        J[i] = 0
        for j in range(1,n):
            KE += 0.5 * masses[j] * np.dot(velocities[j], velocities[j])
            J[i] += masses[j] * (positions[j, 0] * velocities[j, 1] - positions[j, 1] * velocities[j, 0])
            for k in range(j + 1, n):
                r_vec = positions[k] - positions[j]
                r_mag = np.linalg.norm(r_vec)
                v_vec = vel[i-1,j] - vel[i-1,k]

                mu = G * (masses[j] + masses[k])  
                e_vec1 = (J[i-1]/ (mu)) * v_vec[0] - (r_vec[0] / r_mag)
                e_vec2 = -(J[i-1]/ (mu)) * v_vec[1] - (r_vec[1] / r_mag)
                #print(f"e_vec1: {e_vec1}, e_vec2: {e_vec2}")
                e_mag = (e_vec1**2 + e_vec2**2)
                #print(f"emag: {e_mag}")
                e[i-1, j] = e_mag 
                a[i-1, j] = (J[i-1]**2 / (mu)) / (1 - e_mag)
                Log_e[i][j] = np.log10(abs((e_mag - e[0][j])/e[0][j]))
                Log_a[i][j] = np.log10(abs((a[i][j] - a[0][j])/a[0][j]))

                PE -= G * masses[j] * masses[k] / r_mag

                rvec = r_vec / r_mag
                jerk[j] += G * masses[k] * (v_vec / r_mag ** 3 - 3 * np.dot(v_vec, rvec) * rvec / r_mag ** 5)


        E[i] = KE + PE
        Log_E[i] = np.log10(abs((E[i] - E[0])/E[0]))

    return pos,vel,E,J,e,a,at,Log_E,Log_e,Log_a

#Calling the functions for all the integrators
# p_E,v_E,E_E,J_E,e_E,a_E,at_E,t_E,log_E_E,log_e_E,log_a_E = Euler(G, n, masses, positions, velocities, del_t_init, tmax)
# p_EC,v_EC,E_EC,J_EC,e_EC,a_EC,at_EC,t_EC,log_E_EC,log_e_EC,log_a_EC = Euler_Cromer(G, n, masses, positions, velocities, del_t_init, tmax)
# p_LF,v_LF,E_LF,J_LF,e_LF,a_LF,at_LF,t_LF,log_E_LF,log_e_LF,log_a_LF = Leap_frog(G, n, masses, positions, velocities, del_t_init, tmax)
# p_V,v_V,E_V,J_V,e_V,a_V,at_V,t_V,log_E_V,log_e_V,log_a_V = Verlet(G, n, masses, positions, velocities, del_t_init, tmax)
p_RK,v_RK,E_RK,J_RK,e_RK,a_RK,at_RK,log_E_RK,log_e_RK,log_a_RK = RK4(positions, velocities, masses, G, del_t, n)


# '#Plotting the Results
# plt.figure("Trajectories using Different Integrators (10 Orbits)")

# plt.subplot(3, 2, 1)
# for i in range(n):
#     plt.plot(p_E[:, i, 0], p_E[:, i, 1], label=f"Body {i+1}")
#     plt.scatter(positions[:, 0], positions[:, 1], color='black', marker='x', label="Initial Positions")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.title("Euler Integrator")

# plt.subplot(3, 2, 2)
# for i in range(n):
#     plt.plot(p_EC[:, i, 0], p_EC[:, i, 1], label=f"Body {i+1}")
# plt.scatter(positions[:, 0], positions[:, 1], color='black', marker='x', label="Initial Positions")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Euler-Cromer Integrator")

# plt.subplot(3, 2, 3)
# for i in range(n):
#     plt.plot(p_LF[:, i, 0], p_LF[:, i, 1], label=f"Body {i+1}")
# plt.scatter(positions[:, 0], positions[:, 1], color='black', marker='x', label="Initial Positions")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Leap Frog Integrator")

# plt.subplot(3, 2, 4)
# for i in range(n):
#     plt.plot(p_V[:, i, 0], p_V[:, i, 1], label=f"Body {i+1}")
# plt.scatter(positions[:, 0], positions[:, 1], color='black', marker='x', label="Initial Positions")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("Verlet Integrator")

# plt.subplot(3, 2, 5)
# for i in range(n):
#     plt.plot(p_RK[:, i, 0], p_RK[:, i, 1], label=f"Body {i+1}")
# plt.scatter(positions[:, 0], positions[:, 1], color='black', marker='x', label="Initial Positions")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.title("RK Integrator")

# plt.show()

# plt.figure("Energy and Angular Momentum conservation of Different Integrators (10 Orbits)")

# plt.subplot(3,2,1)
# plt.plot(t_E, E_E, label="Energy")
# plt.plot(t_E, J_E, label="Angular Momentum")
# plt.xlabel("t")
# plt.ylabel("E/J")
# plt.legend()
# plt.title("Euler Integrator")
# plt.grid()

# plt.subplot(3,2,2)
# plt.plot(t_EC, E_EC, label="Energy")
# plt.plot(t_EC, J_EC, label="Angular Momentum")
# plt.xlabel("t")
# plt.ylabel("E/J")
# plt.legend()
# plt.title("Euler-Cromer Integrator")
# plt.grid()

# plt.subplot(3,2,3)
# plt.plot(t_LF, E_LF, label="Energy")
# plt.plot(t_LF, J_LF, label="Angular Momentum")
# plt.xlabel("t")
# plt.ylabel("E/J")
# plt.legend()
# plt.title("Leap Frog Integrator")
# plt.grid()

# plt.subplot(3,2,4)
# plt.plot(t_V, E_V, label="Energy")
# plt.plot(t_V, J_V, label="Angular Momentum")
# plt.xlabel("t")
# plt.ylabel("E/J")
# plt.legend()
# plt.title("Verlet Integrator")
# plt.grid()

# plt.subplot(3,2,5)
# plt.plot(t, E_RK, label="Energy")
# plt.plot(t, J_RK, label="Angular Momentum")
# plt.xlabel("t")
# plt.ylabel("E/J")
# plt.legend()
# plt.title("RK Integrator")
# plt.grid()

# plt.show()

# plt.subplot(2,2,1)
# plt.plot(t_E, J_E, label="j-Euler")
# plt.plot(t_EC, J_EC, label="j-Euler Cromer")
# plt.plot(t_LF, J_LF, label="j-Leap Frog")
# plt.plot(t_V, J_V, label="j-Verlet")
# plt.plot(t, J_RK, label="j-RK4")
# plt.xlabel("t")
# plt.ylabel("J")
# plt.xlim(1, 60)
# plt.ylim(0.0008, 0.0012)
# plt.legend()

# plt.subplot(2,2,2)
# plt.plot(t_E, e_E, label="e-Euler")
# plt.plot(t_EC, e_EC, label="e-Euler Cromer")
# plt.plot(t_LF, e_LF, label="e-Leap Frog")
# plt.plot(t_V[0:-1], e_V, label="e-Verlet")
# plt.plot(t, e_RK, label="e-RK4")
# plt.xlabel("t")
# plt.ylabel("e")
# plt.xlim(1, 60)
# plt.ylim(0.9, 1.05)
# plt.legend()

# plt.subplot(2,2,3)
# plt.plot(t_E, a_E, label="a-Euler")
# plt.plot(t_EC, a_EC, label="a-Euler Cromer")
# plt.plot(t_LF, a_LF, label="a-Leap Frog")
# plt.plot(t_V[0:-1], a_V, label="a-Verlet")
# plt.plot(t, a_RK, label="a-RK4")
# plt.xlabel("t")
# plt.ylabel("a")
# plt.xlim(1, 60)
# plt.ylim(0.3, 0.6)
# plt.legend()

# plt.show()

# plt.subplot(2,2,1)
# plt.plot(t_E, log_E_E, label="Euler")
# plt.plot(t_EC, log_E_EC, label="Euler Cromer")
# plt.plot(t_LF, log_E_LF, label="Leap Frog")
# plt.plot(t_V[0:-1], log_E_V, label="Verlet")
# plt.plot(t, log_E_RK, label="RK4")
# plt.xlabel("t")
# plt.ylabel("Relative Error-E")
# plt.ylim(-7.5, 0.5)
# plt.legend()

# plt.subplot(2,2,2)
# plt.plot(t_E, log_e_E, label="Euler")
# plt.plot(t_EC, log_e_EC, label="Euler Cromer")
# plt.plot(t_LF, log_e_LF, label="Leap Frog")
# plt.plot(t_V[0:-1], log_e_V, label="Verlet")
# plt.plot(t, log_e_RK, label="RK4")
# plt.xlabel("t")
# plt.ylabel("Relative Error-e")
# plt.legend()

# plt.subplot(2,2,3)
# plt.plot(t_E, log_a_E, label="Euler")
# plt.plot(t_EC, log_a_EC, label="Euler Cromer")
# plt.plot(t_LF, log_a_LF, label="Leap Frog")
# plt.plot(t_V[0:-1], log_a_V, label="Verlet")
# plt.plot(t, log_a_RK, label="RK4")
# plt.xlabel("t")
# plt.ylabel("Relative Error-a")
# plt.legend()
# plt.tight_layout()

# plt.show()

# plt.figure("Adaptive Time Step")
# plt.plot(at_E, label="Euler Integrator")
# plt.plot(at_EC, label="Euler-Cromer Integrator")
# plt.plot(at_LF, label="Leap Frog Integrator")
# plt.plot(at_V, label="Verlet Integrator")
# plt.plot(at_RK, label="RK Integrator")
# plt.xlabel("Time Step Index")
# plt.ylabel("Adaptive Time Step Size")
# plt.title('Adative Time Step')
# plt.legend()
# plt.show()

for i in range(n):
    plt.plot(p_RK[:, i, 0], p_RK[:, i, 1], label=f"Body {i+1}")
plt.scatter(positions[:, 0], positions[:, 1], color='black', marker='x', label="Initial Positions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("RK Integrator Fixed Time Step")
plt.show()

plt.subplot(1, 2, 1)
plt.plot(t, a_RK[:,0], label="semi-major axis 1")
plt.plot(t, a_RK[:,1], label="semi-major axis 2")
plt.xlabel("t")
plt.ylabel("a")
plt.title("Semi-major axis")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(t, e_RK[:,0], label="eccentricity 1")
plt.plot(t, e_RK[:,1], label="eccentricity 2")
plt.xlabel("t")
plt.ylabel("e")
plt.title("eccentricity")
plt.legend()

plt.show()