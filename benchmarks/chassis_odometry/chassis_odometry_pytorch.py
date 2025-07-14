import torch

@torch.compile(backend="aot_eager")
def chassis_odom_update(
    particles: torch.Tensor,
    odometry: torch.Tensor,
    noise: torch.Tensor,
    max_height: float,
    max_width: float,
) -> torch.Tensor:
    X = 0
    Y = 1

    # Extract deltas and noise std
    x_delta, y_delta = odometry[X], odometry[Y]
    x_std, y_std = noise[X], noise[Y]
    N = particles.shape[0]

    # Generate noise directly on device
    noise_x = torch.normal(0.0, x_std, size=(N,), device=particles.device)
    noise_y = torch.normal(0.0, y_std, size=(N,), device=particles.device)

    # In-place update of particles
    particles[:, X] += x_delta + noise_x
    particles[:, Y] += y_delta + noise_y

    # In-place clamping to bounds
    particles[:, X].clamp_(0.0, max_height)
    particles[:, Y].clamp_(0.0, max_width)

    return particles


def prepare_inputs(particles, odometry, noise, max_height, max_width, device):
    # Convert to PyTorch tensors
    particles_torch = torch.as_tensor(particles, device="cuda" if device == "gpu" else "cpu")
    odometry_torch = torch.as_tensor(odometry, device="cuda" if device == "gpu" else "cpu")
    noise_torch = torch.as_tensor(noise, device="cuda" if device == "gpu" else "cpu")
    
    if device == "gpu":
        torch.cuda.synchronize()
    
    return particles_torch, odometry_torch, noise_torch, max_height, max_width


def run(particles, odometry, noise, max_height, max_width):
    with torch.no_grad():
        out = chassis_odom_update(particles, odometry, noise, max_height, max_width)
    
    if particles.device.type == "cuda":
        torch.cuda.synchronize()
    
    return out 
