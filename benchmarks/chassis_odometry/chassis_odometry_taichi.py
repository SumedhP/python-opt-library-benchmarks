import taichi as ti
import numpy as np

# Constants for indexing
X = 0
Y = 1


@ti.kernel
def chassis_odom_update(
    particles: ti.template(),
    odometry: ti.template(),
    noise: ti.template(),
    max_height: ti.f64,
    max_width: ti.f64,
    out: ti.template(),
):
    N = particles.shape[0]
    
    x_delta = odometry[0]
    y_delta = odometry[1]
    x_std = noise[0]
    y_std = noise[1]
    
    for i in range(N):
        # Generate noise for each particle
        random_noise_x = ti.random() * x_std
        random_noise_y = ti.random() * y_std
        
        # Add movement and noise to each particle
        out[i, X] = particles[i, X] + x_delta + random_noise_x
        out[i, Y] = particles[i, Y] + y_delta + random_noise_y
        
        # Clip particles to the map boundaries
        if out[i, X] < 0:
            out[i, X] = 0
        elif out[i, X] > max_height:
            out[i, X] = max_height
            
        if out[i, Y] < 0:
            out[i, Y] = 0
        elif out[i, Y] > max_width:
            out[i, Y] = max_width


def prepare_inputs(particles, odometry, noise, max_height, max_width, device):
    # Create Taichi fields
    particles_field = ti.field(dtype=ti.f64, shape=particles.shape)
    odometry_field = ti.field(dtype=ti.f64, shape=odometry.shape)
    noise_field = ti.field(dtype=ti.f64, shape=noise.shape)
    out_field = ti.field(dtype=ti.f64, shape=particles.shape)
    
    # Copy data to Taichi fields
    particles_field.from_numpy(particles)
    odometry_field.from_numpy(odometry)
    noise_field.from_numpy(noise)
    out_field.fill(0)
    
    return particles_field, odometry_field, noise_field, max_height, max_width, out_field


def run(particles, odometry, noise, max_height, max_width, out):
    chassis_odom_update(particles, odometry, noise, max_height, max_width, out)
    ti.sync()
    return out 
