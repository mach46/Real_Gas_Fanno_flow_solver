import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

fluid = "Methane"

"""----------Real gas properties----------"""
def get_properties_PT(P, T):
    """
    Inputs:
        P, T
    Returns:
        rho, h, a
    """
    rho = PropsSI('D', 'P', P, 'T', T, fluid)
    h   = PropsSI('H', 'P', P, 'T', T, fluid)
    a   = PropsSI('A', 'P', P, 'T', T, fluid)
    mu = PropsSI('V', 'P', P, 'T', T, fluid)
    Cp = PropsSI('Cpmass', 'P', P, 'T', T, fluid)
    Cv = PropsSI('Cvmass', 'P', P, 'T', T, fluid)
    gamma = Cp / Cv
    return rho, h, a, mu, gamma

def get_properties_HP(h, P):
    """
    Inputs:
        h, P
    Returns:
        T, rho, a
    """
    T   = PropsSI('T', 'H', h, 'P', P, fluid)
    rho = PropsSI('D', 'H', h, 'P', P, fluid)
    a   = PropsSI('A', 'H', h, 'P', P, fluid)
    mu = PropsSI('V', 'H', h, 'P', P, fluid)
    Cp = PropsSI('Cpmass', 'H', h, 'P', P, fluid)
    Cv = PropsSI('Cvmass', 'H', h, 'P', P, fluid)
    gamma = Cp / Cv
    return T, rho, a, mu, gamma

""" Reynolds number """
def reynolds_number(rho, u, D, mu):
    return rho * u * D / mu

""" Friction factor """
def colebrook_residual(f, Re, D, epsilon):
    return 1 / np.sqrt(f) + 2.0 * np.log10(
        (epsilon / (3.7 * D)) + (2.51 / (Re * np.sqrt(f)))
    )

def friction_factor_colebrook(Re, D, epsilon):
    if Re < 2300:
        return 64 / Re
    # initial guess (Blasius)
    f_guess = np.array([0.02, 0.02, 0.02, 0.02])
    func = lambda f: colebrook_residual(f, Re, D, epsilon)
    f_solution = fsolve(func, f_guess)[0]
    return f_solution

def friction_factor_haaland(Re, D, epsilon):
    # Laminar
    if Re < 2300:
        return 64 / Re

    # Haaland (Darcy)
    term = (epsilon / (3.7 * D)) ** 1.11 + 6.9 / Re
    f = (-1.8 * np.log10(term)) ** (-2)
    return f

""" Continuity Momentum and Energy equations """
def velocity_from_continuity(m_dot, rho, A):
    return m_dot / (rho * A)

def momentum_update(P_i, rho_i, u_i, u_next, f, Dh, dx):
    return P_i \
           - rho_i * u_i * (u_next - u_i) \
           - (f / (2 * Dh)) * rho_i * u_i**2 * dx

def energy_update(h0, u_next):
    return h0 - 0.5 * u_next**2

""" Residual """
def residual(u_next, P_i, rho_i, u_i, h0, f, Dh, dx, m_dot, A):
    # Energy
    h_next = energy_update(h0, u_next)

    # Momentum
    P_next = momentum_update(P_i, rho_i, u_i, u_next, f, Dh, dx)

    # EOS
    T_next, rho_next, a_next, mu_next, gamma_next = get_properties_HP(h_next, P_next)

    # Continuity
    u_calc = velocity_from_continuity(m_dot, rho_next, A)

    # Residual
    return u_next - u_calc

""" solution of u_i+1 """
def solve_velocity(u_i, P_i, rho_i, h0, f, Dh, dx, m_dot, A):
    func = lambda u: residual(u, P_i, rho_i, u_i, h0, f, Dh, dx, m_dot, A)

    u_next = fsolve(func, u_i)[0]

    return u_next


def update_state(Pt_i, P_i, T_i, rho_i, h_i, u_i,
                 u_next, f, Dh, dx):
    # Total enthalpy
    h0 = h_i + 0.5 * u_i ** 2

    # Energy
    h_next = energy_update(h0, u_next)

    # Momentum
    P_next = momentum_update(P_i, rho_i, u_i, u_next, f, Dh, dx)

    # EOS
    T_next, rho_next, a_next, mu_next, gamma_next = get_properties_HP(h_next, P_next)

    # Mach number
    M_next = u_next / a_next

    # Total Pressure
    Pt_next = P_next * (1 + (gamma_next - 1) / 2 * M_next ** 2) ** (gamma_next / (gamma_next - 1))

    return Pt_next, P_next, T_next, rho_next, h_next, a_next



""" Input Parameters """

P1 = 60e5     # Pressure [Pa]
T1 = 250      # Temperature [K]

#f = 0.015     # Darcy friction factor
epsilon = 0

d_array = np.array([0.0020, 0.0030, 0.0040, 0.0050])  # [m]
m_dot_array = np.array([0.016, 0.036, 0.064, 0.100])  # [kg/s]

dx = 1e-3        # Step size [m]
max_steps = 100000

M_target = 0.8

"""GLOBAL STORAGE (ALL CASES)"""
L_results = []
x_profiles = []
M_profiles = []
P_profiles = []
T_profiles = []
Pt_profiles = []
f_profiles = []

""" Outer loop """
for d, m_dot in zip(d_array, m_dot_array):

    """ Initialization """
    Dh = d
    A  = np.pi * d**2 / 4

    P = P1
    T = T1

    rho, h, a, mu, gamma = get_properties_PT(P, T)
    rho1 = rho  # store inlet density

    u = m_dot / (rho * A)
    u1 = u  # Store Inlet velocity

    M = u / a
    M1 = M  # Store Inlet mach number


    #Total pressure
    Pt = P * (1 + (gamma - 1) / 2 * M ** 2) ** (gamma / (gamma - 1))
    Pt1 = Pt    # Store inlet Stagnation Pressure

    # Friction factor
    Re = reynolds_number(rho, u, Dh, mu)
    f = friction_factor_haaland(Re, Dh, epsilon)

    x = 0.0

    x_profile = []
    M_profile = []
    P_profile = []
    T_profile = []
    Pt_profile = []
    f_profile = []

    # store initial point
    x_profile.append(x)
    M_profile.append(M)
    P_profile.append(P)
    T_profile.append(T)
    Pt_profile.append(Pt)
    f_profile.append(f)

    """ Inner Marching loop """
    for step in range(max_steps):

        # STORE CURRENT STATE
        x_profile.append(x)
        M_profile.append(M)
        P_profile.append(P)
        T_profile.append(T)
        Pt_profile.append(Pt)
        f_profile.append(f)


        # STOP CONDITION
        if M >= M_target:
            break


        # TOTAL ENTHALPY
        h0 = h + 0.5 * u ** 2

        Re = reynolds_number(rho, u, Dh, mu)
        f = friction_factor_haaland(Re, Dh, epsilon)


        # SOLVE FOR NEXT VELOCITY
        u_next = solve_velocity(u, P, rho, h0, f, Dh, dx, m_dot, A)


        # UPDATE STATE (energy + momentum + EOS)
        Pt, P, T, rho, h, a = update_state(
            Pt, P, T, rho, h, u,
            u_next, f, Dh, dx
        )

        # UPDATE FLOW VARIABLES
        u = u_next
        M = u / a

        # UPDATE POSITION
        x += dx


    """ Final results(per case) """
    L_results.append(x)
    x_profiles.append(x_profile)
    M_profiles.append(M_profile)
    P_profiles.append(P_profile)
    T_profiles.append(T_profile)
    f_profiles.append(f_profile)
    Pt_profiles.append(Pt_profile)


    """ Results """
    print("\n=======================================================")
    print(f"D = {d:.4f} m | m_dot = {m_dot:.4f} kg/s")
    print("-------------------------------------------------------")
    print(f"Length (L)                      : {x:.6f} m")
    print(f"Steps taken                     : {len(x_profile)}")
    print(f"Stagnation Pressure Drop        : {(Pt1-Pt):.6f} Pa")
    print(f"Static Pressure Drop            : {(P1-P):.6f} Pa")
    print("-------------------------------------------------------")
    print(f"Inlet Mach                      : {M1:.6f}")
    print(f"Inlet Velocity                  : {u1:.6f} m/s")
    print(f"Inlet Stagnation Pressure       : {Pt1:.6f} Pa")
    print(f"Inlet Static Pressure           : {P1:.6f} Pa")
    print(f"Inlet Static Temperature        : {T1:.6f} K")
    print(f"Inlet Density                   : {rho1:.6f} kg/m3")
    print("-------------------------------------------------------")
    print(f"Outlet Mach                     : {M:.6f}")
    print(f"Outlet Velocity                 : {u:.6f} m/s")
    print(f"Outlet Stagnation Pressure      : {Pt:.6f} Pa")
    print(f"Outlet Static Pressure          : {P:.6f} Pa")
    print(f"Outlet Static Temperature       : {T:.6f} K")
    print(f"Outlet Density                  : {rho:.6f} kg/m3")
    print("=======================================================")





    """ Mach number graph """
    plt.figure()

    for i in range(len(x_profiles)):
        plt.plot(
            x_profiles[i],
            M_profiles[i],
            label=f'd={d_array[i] * 1000:.1f} mm, m={m_dot_array[i]:.3f}'
        )

    plt.xlabel("Length (m)")
    plt.ylabel("Mach Number")
    plt.title("Mach Number Variation Along Channel")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    """ Total Pressure graph """
    plt.figure()

    for i in range(len(x_profiles)):
        plt.plot(
            x_profiles[i],
            Pt_profiles[i],
            label=f'd={d_array[i] * 1000:.1f} mm, m={m_dot_array[i]:.3f}'
        )

    plt.xlabel("Length (m)")
    plt.ylabel("Total Pressure (Pa)")
    plt.title("Total Pressure Drop Along Channel")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    """ Static Pressure graph """
    plt.figure()

    for i in range(len(x_profiles)):
        plt.plot(
            x_profiles[i],
            P_profiles[i],
            label=f'd={d_array[i] * 1000:.1f} mm'
        )

    plt.xlabel("Length (m)")
    plt.ylabel("Static Pressure (Pa)")
    plt.title("Static Pressure Variation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    """ Static Temperature graph """
    plt.figure()

    for i in range(len(x_profiles)):
        plt.plot(
            x_profiles[i],
            T_profiles[i],
            label=f'd={d_array[i] * 1000:.1f} mm'
        )

    plt.xlabel("Length (m)")
    plt.ylabel("Static Temperature (Pa)")
    plt.title("Static Temperature Variation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    """ Friction factor graph """
    plt.figure()

    for i in range(len(x_profiles)):
        plt.plot(
            x_profiles[i],
            f_profiles[i],
            label=f'd={d_array[i] * 1000:.1f} mm'
        )

    plt.xlabel("Length (m)")
    plt.ylabel("Friction Factor")
    plt.title("Friction Factor Variation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()