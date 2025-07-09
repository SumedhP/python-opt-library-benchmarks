import torch

# Constants for indexing
X = 0
Y = 1


@torch.jit.script
def chassis_odom_update(
    particles: torch.Tensor,
    odometry: torch.Tensor,
    noise: torch.Tensor,
    max_height: float,
    max_width: float,
) -> torch.Tensor:
    N = particles.shape[0]
    
    x_delta = odometry[0]
    y_delta = odometry[1]
    x_std = noise[0]
    y_std = noise[1]
    
    # Generate noise for all particles at once
    random_noise_x = torch.normal(torch.zeros(N, device=particles.device), x_std)
    random_noise_y = torch.normal(torch.zeros(N, device=particles.device), y_std)
    
    # Vectorized update
    new_particles = particles.clone()
    new_particles[:, 0] += x_delta + random_noise_x
    new_particles[:, 1] += y_delta + random_noise_y
    
    # Vectorized clipping
    new_particles[:, 0] = torch.clamp(new_particles[:, 0], 0, max_height)
    new_particles[:, 1] = torch.clamp(new_particles[:, 1], 0, max_width)
    
    return new_particles


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
