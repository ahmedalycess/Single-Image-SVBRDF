import torch
import math
from PIL import Image
import os

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
    current_view_pos = generate_normalized_random_direction(batch_size)                # Shape: [batch_size, 3]
    current_light_pos = generate_normalized_random_direction(batch_size)               # Shape: [batch_size, 3]
    wi = current_light_pos.unsqueeze(2).unsqueeze(3)                                   # Shape: [batch_size, 3, 1, 1]
    wo = current_view_pos.unsqueeze(2).unsqueeze(3)                                    # Shape: [batch_size, 3, 1, 1]
    rendered_diffuse = render_fn(targets, wi, wo)          # rendered_diffuse[0]:results.shape: [batch_size, 3, 256, 256])
    rendered_diffuse_outputs = render_fn(outputs, wi, wo)  # rendered_outputs[0]:results.shape: [batch_size, 3, 256, 256])
    return [rendered_diffuse, rendered_diffuse_outputs]

def generate_specular_rendering(batch_size, surface_array, targets, outputs, render_fn, include_diffuse):
    current_view_dir = generate_normalized_random_direction(batch_size)         # Shape: [batch_size, 3]
    current_light_dir = current_view_dir * torch.tensor([-1.0, -1.0, 1.0])      # Shape: [batch_size, 3]
                                                                                # shape: [batch_size, 3]
    current_shift = torch.cat([torch.rand(batch_size, 2) * 2 - 1, torch.zeros(batch_size, 1) + 0.0001], dim=-1)
    def generate_distance(batch_size):
        gaussian = torch.normal(mean=0.5, std=0.75, size=(batch_size, 1))  # Gaussian distribution
        return torch.exp(gaussian)  # Exponential transformation

    current_view_pos = current_view_dir * generate_distance(batch_size=batch_size) + current_shift # Shape: [batch_size, 3]
    current_light_pos = current_light_dir * generate_distance(batch_size=batch_size) + current_shift # Shape: [batch_size, 3]

    current_view_pos = current_view_pos.unsqueeze(2).unsqueeze(3)   # Shape: [batch_size, 3, 1, 1]
    current_light_pos = current_light_pos.unsqueeze(2).unsqueeze(3) # Shape: [batch_size, 3, 1, 1]


    wo = current_view_pos - surface_array # Shape: [batch_size, 3, 1, 1]
    wi = current_light_pos - surface_array # Shape: [batch_size, 3, 1, 1]
    rendered_specular = render_fn(targets, wi, wo, include_diffuse=include_diffuse)   # rendered_specular[0]:results.shape: [batch_size, 3, 256, 256])
    rendered_specular_outputs = render_fn(outputs, wi, wo, include_diffuse=include_diffuse) # rendered_outputs[0]:results.shape: [batch_size, 3, 256, 256])

    # # save the last images
    # if not os.path.exists("images"):
    #     os.makedirs("images")

    # for i in range(batch_size):
    #     img = Image.fromarray((rendered_specular[0][i].detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0))
    #     img.save(os.path.join("images", "target_" + str(i) + ".png"))

    #     img = Image.fromarray((rendered_specular_outputs[0][i].detach().cpu().numpy() * 255).astype('uint8').transpose(1, 2, 0))
    #     img.save(os.path.join("images", "output_" + str(i) + ".png"))

    return [rendered_specular, rendered_specular_outputs]