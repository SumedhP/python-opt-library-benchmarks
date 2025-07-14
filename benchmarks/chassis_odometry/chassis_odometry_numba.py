import numpy as np
import numba as nb

# Constants for indexing
X = 0
Y = 1


@nb.njit(parallel=True, fastmath=True, boundscheck=False, cache=True)
def chassis_odom_update(
    particles: np.ndarray,
    odometry: np.ndarray,
    noise: np.ndarray,
    max_height: float,
    max_width: float,
) -> None:
    N = particles.shape[0]
    
    x_delta = odometry[X]
    y_delta = odometry[Y]
    x_std = noise[X]
    y_std = noise[Y]
    
    # Generate noise for all particles at once
    random_noise_x = np.random.normal(0, x_std, N)
    random_noise_y = np.random.normal(0, y_std, N)
    
    # Parallel loop for updating particles
    for i in nb.prange(N):
        # Add movement and noise to each particle
        particles[i, X] += x_delta + random_noise_x[i]
        particles[i, Y] += y_delta + random_noise_y[i]
        
        # Clip particles to the map boundaries
        if particles[i, X] < 0:
            particles[i, X] = 0
        elif particles[i, X] > max_height:
            particles[i, X] = max_height
            
        if particles[i, Y] < 0:
            particles[i, Y] = 0
        elif particles[i, Y] > max_width:
            particles[i, Y] = max_width


def run(particles, odometry, noise, max_height, max_width):
    chassis_odom_update(particles, odometry, noise, max_height, max_width)
    return particles 
