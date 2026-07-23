"""Gauss equation; Jacobi Solver"""
import numpy as np
import time
from tqdm import trange
import matplotlib.pyplot as plt

# constants
#epsilon_0 = 8.854187817e-12                                                            # vacuum permittivity in F/m
#electron_charge = 1.602176634e-19                                                      # elementary charge in C. Take positive sign

# geometry and discretization
height, width = 1, 1
dx, dy = 0.0025, 0.0025
x0, y0 = -width / 2, -height / 2

# boundary conditions
bottom_boundary = 0.8
top_boundary = 0.8
left_boundary = 0
right_boundary = 0

# charge distribution
charge_radius = 0.1
point_charges_positions= np.array([[0.3, 0.3], [0.7, 0.7]]) + np.array([[x0, y0]])      # dim: [2, 2] + [1, 2] -> [2, 2]
charges = np.array([2, -2])                                                             # charge to epsilon ratio instead of charge. 
rho = None                                                                              # charge density array, initialized to None

# hyperparameters
iterations = 100001
start_time = time.perf_counter()
if dx != dy:
    raise ValueError("dx and dy must be equal for this implementation.")

def boundary_conditions(nx, ny):                                                    # interval x, interval y, u(x,y)
    """Set up boundary conditions for the potential field."""
    u = np.zeros((ny, nx))                                                          # y and x are reversed due to the array indexing;

    # Remeber the coorinate of y is up-side down
    u[0, :] = bottom_boundary       # real bottom boundary
    u[-1, :] = top_boundary         # real top boundary
    u[:, 0] = left_boundary         # left boundary
    u[:, -1] = right_boundary       # right boundary

    return u

def charge_density(nx, ny, point_charges_positions, charges, charge_radius):
    rho = np.zeros((ny, nx))  # charge density array
    charges_density = charges / (np.pi * charge_radius ** 2)

    point_charges = np.array([[positions[0], positions[1], charge_density] for positions, charge_density in zip(point_charges_positions, charges_density)])
    for point_charge in point_charges:
        # i and j are following the indexing of the array, which is different from the x-y coordinate system
        for i in range(ny):
            for j in range(nx):
                r = np.sqrt((x0 + j * dx - point_charge[0])**2 + (y0 + i * dy - point_charge[1])**2)
                if r <= charge_radius:  # include a small tolerance to account for numerical errors
                    rho[i, j] = point_charge[-1]
    
    return rho

def Jacobi(u, max_iterations, rho=False):
    """Perform Jacobi iterations to solve the Laplace equation."""
    #ny, nx = u.shape
    u_new = u.copy()
    for iteration in trange(max_iterations, desc="Jacobi", unit="iter"):
        if rho is False:
            u_new[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] +u[:-2, 1:-1] +u[1:-1, 2:] +u[1:-1, :-2])
        else:
            u_new[1:-1, 1:-1] = 0.25 * ((u[2:, 1:-1] +u[:-2, 1:-1] +u[1:-1, 2:] +u[1:-1, :-2]) + (dx**2) * rho[1:-1, 1:-1]) #/ epsilon_0)   # requiring dx = dy, charge to epsilon ratio instead.
        u = u_new
    print(f'Final iteration: {iteration}, Residual: {np.linalg.norm(u_new - u)}')

    return u

def main():
    nx = int(width / dx + 1)
    ny = int(height / dy + 1)
    print (nx, ny) 
    v = boundary_conditions(nx, ny)
    rho = charge_density(nx, ny, point_charges_positions, charges, charge_radius)
    v = Jacobi(v, iterations, rho)    # u, max_iterations, rho, Gaussian unit per electron charge

    x = np.linspace(x0, x0 + width, nx)
    y = np.linspace(y0, y0 + height, ny)
    X, Y = np.meshgrid(x, y)

    # Graph of electric potential
    plt.contourf(X, Y, v, levels=50, cmap='viridis')
    plt.colorbar(format='%.3g')
    plt.title('v')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("2point_charges-Jacobi.png", dpi=600)
    #plt.show()
    plt.close()
    
    structure_name = "2point_charges-Jacobi.csv"
    with open(structure_name, "w") as f:
        f.write("v, x, y\n")
        for i in range(int(ny)):
            for j in range(int(nx)):
                f.write(f"{v[i, j]}, {x[j]}, {y[i]}\n")

    # Graph of charge density
    if rho is not None:
        plt.contourf(X, Y, rho, levels=50, cmap='viridis')
        plt.colorbar(format='%.3g')
        plt.title('rho')
        plt.xlabel('x')
        plt.ylabel('y')
        # plt.savefig("2point_charges-rho.png", dpi=600)
        plt.show()
        plt.close()

        charges_name = "2point_charges-rho.csv"
        with open(charges_name, "w") as f:
            f.write("rho, x, y\n")
            for i in range(int(ny)):
                for j in range(int(nx)):
                    f.write(f"{rho[i, j]}, {x[j]}, {y[i]}\n")

if __name__ == "__main__":
    main()

