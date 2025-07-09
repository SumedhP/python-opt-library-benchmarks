"""
==========================================================================
  Chassis Odometry Update with Gaussian Noise
  Particle filter odometry update with noise simulation
==========================================================================
"""

import numpy as np

# Constants for indexing
X = 0
Y = 1


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
    
    # Vectorized update
    particles[:, X] += x_delta + random_noise_x
    particles[:, Y] += y_delta + random_noise_y
    
    # Vectorized clipping
    particles[:, X] = np.clip(particles[:, X], 0, max_height)
    particles[:, Y] = np.clip(particles[:, Y], 0, max_width)


def run(particles, odometry, noise, max_height, max_width):
    chassis_odom_update(particles, odometry, noise, max_height, max_width)
    return particles 
