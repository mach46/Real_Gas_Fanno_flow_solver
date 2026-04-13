import numpy as np
from scipy.optimize import fsolve
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
    return rho, h, a, mu

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
    return T, rho, a

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
    f_guess = 0.02
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
    T_next, rho_next, a_next = get_properties_HP(h_next, P_next)

    # Continuity
    u_calc = velocity_from_continuity(m_dot, rho_next, A)

    # Residual
    return u_next - u_calc

""" solution of u_i+1 """
def solve_velocity(u_i, P_i, rho_i, h0, f, Dh, dx, m_dot, A):
    func = lambda u: residual(u, P_i, rho_i, u_i, h0, f, Dh, dx, m_dot, A)

    u_next = fsolve(func, u_i)[0]

    return u_next


def update_state(P_i, T_i, rho_i, h_i, u_i,
                 u_next, f, Dh, dx):
    # Total enthalpy
    h0 = h_i + 0.5 * u_i ** 2

    # Energy
    h_next = energy_update(h0, u_next)

    # Momentum
    P_next = momentum_update(P_i, rho_i, u_i, u_next, f, Dh, dx)

    # EOS
    T_next, rho_next, a_next = get_properties_HP(h_next, P_next)

    return P_next, T_next, rho_next, h_next, a_next



""" Input Parameters """

P0 = 60e5     # Pressure [Pa]
T0 = 250     # Temperature [K]

#f = 0.015     # Darcy friction factor
epsilon = 0

d_array = np.array([0.0020, 0.0030, 0.0040, 0.0050])  # [m]
m_dot_array = np.array([0.016, 0.036, 0.064, 0.100])  # [kg/s]

dx = 1e-3        # Step size [m]
max_steps = 100000

M_target = 0.5

"""GLOBAL STORAGE (ALL CASES)"""
L_results = []
x_profiles = []
M_profiles = []
P_profiles = []
T_profiles = []

""" Outer loop """
for d, m_dot in zip(d_array, m_dot_array):

    """ Initialization """
    Dh = d
    A  = np.pi * d**2 / 4

    P = P0
    T = T0

    rho, h, a, mu = get_properties_PT(P, T)

    u = m_dot / (rho * A)

    M = u / a

    x = 0.0

    x_profile = []
    M_profile = []
    P_profile = []
    T_profile = []

    # store initial point
    x_profile.append(x)
    M_profile.append(M)
    P_profile.append(P)
    T_profile.append(T)

    """ Inner Marching loop """
    for step in range(max_steps):

        # STORE CURRENT STATE
        x_profile.append(x)
        M_profile.append(M)
        P_profile.append(P)
        T_profile.append(T)


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
        P, T, rho, h, a = update_state(
            P, T, rho, h, u,
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


    """ Results """
    print("\n==============================")
    print(f"D = {d:.4f} m | m_dot = {m_dot:.4f} kg/s")
    print("------------------------------")
    print(f"Length (L)   : {x:.6f} m")
    print(f"Final Mach   : {M:.6f}")
    print(f"Steps taken  : {len(x_profile)}")
    print("==============================")