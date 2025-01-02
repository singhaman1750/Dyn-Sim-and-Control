import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

class SimplePendulum:
    def __init__(self, length, mass, damping_coefficient, gravity=9.81):
        """
        Initialize the pendulum parameters.

        Parameters:
        - length (float): Length of the pendulum (m)
        - mass (float): Mass of the pendulum bob (kg)
        - damping_coefficient (float): Viscous friction coefficient (N*m*s/rad)
        - gravity (float): Acceleration due to gravity (default: 9.81 m/s^2)
        """
        self.l = length
        self.m = mass
        self.c = damping_coefficient
        self.g = gravity

    def equation_of_motion(self, t, y):
        """
        Defines the equations of motion for the pendulum.

        Parameters:
        - t (float): Time (s)
        - y (list): State vector [theta, theta_dot]

        Returns:
        - dydt (list): Time derivatives [theta_dot, theta_double_dot]
        """
        theta, theta_dot = y
        theta_double_dot = -(self.g / self.l) * np.sin(theta) - (self.c / (self.m * self.l**2)) * theta_dot
        return [theta_dot, theta_double_dot]

    def simulate(self, theta0, theta_dot0, t_span, t_eval=None):
        """
        Simulates the pendulum's motion.

        Parameters:
        - theta0 (float): Initial angle (rad)
        - theta_dot0 (float): Initial angular velocity (rad/s)
        - t_span (tuple): Time span for simulation (start, end) in seconds
        - t_eval (array): Time points where the solution is desired

        Returns:
        - sol (OdeResult): Solution object from `scipy.integrate.solve_ivp`
        """
        y0 = [theta0, theta_dot0]
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 1000)

        sol = solve_ivp(self.equation_of_motion, t_span, y0, t_eval=t_eval, method='RK45')
        return sol

    def plot_simulation(self, sol):
        """
        Plots the results of the simulation.

        Parameters:
        - sol (OdeResult): Solution object from `scipy.integrate.solve_ivp`
        """
        plt.figure(figsize=(10, 5))
        plt.plot(sol.t, sol.y[0], label='Theta (rad)')
        plt.plot(sol.t, sol.y[1], label='Angular Velocity (rad/s)')
        plt.title('Pendulum Simulation')
        plt.xlabel('Time (s)')
        plt.ylabel('State')
        plt.legend()
        plt.grid()
        plt.show()

    def animate_simulation(self, sol):
        """
        Animates the pendulum's motion.

        Parameters:
        - sol (OdeResult): Solution object from `scipy.integrate.solve_ivp`
        """
        # Extract angle data
        theta = sol.y[0]

        # Calculate pendulum position
        x = self.l * np.sin(theta)
        y = -self.l * np.cos(theta)

        # Set up the figure and axis
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-self.l - 0.1, self.l + 0.1)
        ax.set_ylim(-self.l - 0.1, self.l + 0.1)
        ax.set_aspect('equal')
        ax.grid()

        # Create the pendulum line and bob
        line, = ax.plot([], [], 'o-', lw=2)

        # Initialize the animation
        def init():
            line.set_data([], [])
            return line,

        # Update function for animation
        def update(frame):
            line.set_data([0, x[frame]], [0, y[frame]])
            return line,

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True, interval=20)

        plt.title('Pendulum Animation')
        plt.show()

# Example Usage
if __name__ == "__main__":
    # Define pendulum parameters
    length = 0.2  # meters
    mass = 1.0    # kg
    damping_coefficient = 0.1  # N*m*s/rad

    # Create pendulum object
    pendulum = SimplePendulum(length, mass, damping_coefficient)

    # Initial conditions and simulation time
    theta0 = np.pi/2 # Initial angle (45 degrees in radians)
    theta_dot0 = 0.0    # Initial angular velocity (rad/s)
    t_span = (0, 10)    # Simulate for 10 seconds

    # Run simulation
    sol = pendulum.simulate(theta0, theta_dot0, t_span)

    # Animate results
    pendulum.animate_simulation(sol)

    # Plot results
    pendulum.plot_simulation(sol)


