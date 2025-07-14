import importlib
import functools


def generate_inputs(size):
    import numpy as np

    np.random.seed(17)

    # Number of particles
    N = size
    
    # Generate particle positions (N, 2) - [x, y]
    particles = np.random.uniform(0, 100, size=(N, 2)).astype(np.float32)
    
    # Generate odometry update [dx, dy]
    odometry = np.random.uniform(-1, 1, size=2).astype(np.float32)
    
    # Generate noise standard deviations [std_x, std_y]
    noise = np.random.uniform(0.1, 0.5, size=2).astype(np.float32)
    
    # Map boundaries
    max_height = 100.0
    max_width = 100.0

    return particles, odometry, noise, max_height, max_width


def try_import(backend):
    try:
        return importlib.import_module(f".chassis_odometry_{backend}", __name__)
    except ImportError:
        return None


def get_callable(backend, size, device="cpu"):
    backend_module = try_import(backend)
    inputs = generate_inputs(size)
    if hasattr(backend_module, "prepare_inputs"):
        inputs = backend_module.prepare_inputs(*inputs, device=device)
    return functools.partial(backend_module.run, *inputs)


__implementations__ = (
    "numba", 
    "jax",
    "numpy",
    "pytorch",
) 
