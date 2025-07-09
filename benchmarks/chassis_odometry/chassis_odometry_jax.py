import jax
import jax.numpy as np

# Constants for indexing
X = 0
Y = 1


@jax.jit
def chassis_odom_update(
    particles: np.ndarray,
    odometry: np.ndarray,
    noise: np.ndarray,
    max_height: float,
    max_width: float,
    key: jax.random.PRNGKey,
) -> np.ndarray:
    N = particles.shape[0]
    
    x_delta = odometry[0]
    y_delta = odometry[1]
    x_std = noise[0]
    y_std = noise[1]
    
    # Generate noise for all particles at once
    key_x, key_y = jax.random.split(key)
    random_noise_x = jax.random.normal(key_x, shape=(N,)) * x_std
    random_noise_y = jax.random.normal(key_y, shape=(N,)) * y_std
    
    # Vectorized update
    new_particles = particles.at[:, X].add(x_delta + random_noise_x)
    new_particles = new_particles.at[:, Y].add(y_delta + random_noise_y)
    
    # Vectorized clipping
    new_particles = new_particles.at[:, X].set(
        np.clip(new_particles[:, X], 0, max_height)
    )
    new_particles = new_particles.at[:, Y].set(
        np.clip(new_particles[:, Y], 0, max_width)
    )
    
    return new_particles


def prepare_inputs(particles, odometry, noise, max_height, max_width, device):
    # Convert to JAX arrays
    particles_jax = np.array(particles)
    odometry_jax = np.array(odometry)
    noise_jax = np.array(noise)
    max_height_jax = np.array(max_height)
    max_width_jax = np.array(max_width)
    
    # Create a random key for JAX
    key = jax.random.PRNGKey(42)
    
    return particles_jax, odometry_jax, noise_jax, max_height_jax, max_width_jax, key


def run(particles, odometry, noise, max_height, max_width, key):
    out = chassis_odom_update(particles, odometry, noise, max_height, max_width, key)
    out.block_until_ready()
    return out 
