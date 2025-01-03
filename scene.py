import torch
import torch.nn.functional as F
import math

def generate_normalized_random_direction(batch_size, low_eps=0.001, high_eps=0.05):
    r1 = torch.rand(batch_size, 1) * (1.0 - low_eps - high_eps) + low_eps
    r2 = torch.rand(batch_size, 1)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2

    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - r**2)
    final_vec = torch.cat([x, y, z], dim=-1)  # Shape: [batch_size, 3]
    return final_vec

def generate_diffuse_rendering(batch_size, targets, outputs, render_fn):
    current_view_pos = generate_normalized_random_direction(batch_size)
    current_light_pos = generate_normalized_random_direction(batch_size)
    wi = current_light_pos.unsqueeze(1).unsqueeze(1)
    wo = current_view_pos.unsqueeze(1).unsqueeze(1)
    rendered_diffuse = render_fn(targets, wi, wo)
    rendered_diffuse_outputs = render_fn(outputs, wi, wo)

    return [rendered_diffuse, rendered_diffuse_outputs]

def generate_specular_rendering(batch_size, surface_array, targets, outputs, render_fn, include_diffuse):
    current_view_dir = generate_normalized_random_direction(batch_size)
    current_light_dir = current_view_dir * torch.tensor([-1.0, -1.0, 1.0])

    current_shift = torch.cat([torch.rand(batch_size, 2) * 2 - 1, torch.zeros(batch_size, 1) + 0.0001], dim=-1)

    def generate_distance(batch_size):
        gaussian = torch.normal(mean=0.5, std=0.75, size=(batch_size, 1))  # Gaussian distribution
        return torch.exp(gaussian)  # Exponential transformation

    current_view_pos = current_view_dir * generate_distance(batch_size=batch_size) + current_shift
    current_light_pos = current_light_dir * generate_distance(batch_size=batch_size) + current_shift

    current_view_pos = current_view_pos.unsqueeze(1).unsqueeze(1)
    current_light_pos = current_light_pos.unsqueeze(1).unsqueeze(1)

    wo = current_view_pos - surface_array
    wi = current_light_pos - surface_array
    rendered_specular = render_fn(targets, wi, wo, include_diffuse=include_diffuse)
    rendered_specular_outputs = render_fn(outputs, wi, wo, include_diffuse=include_diffuse)

    return [rendered_specular, rendered_specular_outputs]