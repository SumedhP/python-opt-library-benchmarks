import jax
import jax.numpy as jnp

# Constants for indexing
X = 0
Y = 1


@jax.jit
def chassis_odom_update(
    particles: jnp.ndarray,
    odometry: jnp.ndarray,
    noise_std: jnp.ndarray,
    max_height: float,
    max_width: float,
    key: jax.random.PRNGKey,
) -> jnp.ndarray:
    # Split key only once
    key, subkey = jax.random.split(key)
    
    # Draw 2-D Gaussian noise in one shot
    # shape = (N,2), stds broadcast across rows
    noise = jax.random.normal(subkey, shape=particles.shape) * noise_std

    # Broadcast odometry [2] to [N,2]
    delta = odometry  # shape (2,) will broadcast

    # Single fused update + clamp
    updated = particles + delta + noise

    # Clip each column
    # Note: jnp.clip is elementwise, but this is still one XLA fusion group
    updated = updated.at[:, 0].set(jnp.clip(updated[:, 0], 0.0, max_height))
    updated = updated.at[:, 1].set(jnp.clip(updated[:, 1], 0.0, max_width))

    return updated


def prepare_inputs(particles, odometry, noise, max_height, max_width, device):
    # Convert to JAX arrays
    particles_jax = jnp.array(particles)
    odometry_jax = jnp.array(odometry)
    noise_jax = jnp.array(noise)
    max_height_jax = jnp.array(max_height)
    max_width_jax = jnp.array(max_width)
    
    # Create a random key for JAX
    key = jax.random.PRNGKey(42)
    
    return particles_jax, odometry_jax, noise_jax, max_height_jax, max_width_jax, key


def run(particles, odometry, noise, max_height, max_width, key):
    out = chassis_odom_update(particles, odometry, noise, max_height, max_width, key)
    out.block_until_ready()
    return out 
