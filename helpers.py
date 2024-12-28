import torch

# ------------------------
# Helpers
# ------------------------

def torch_normalize(tensor, eps=1e-8):
    """
    Normalize the tensor along the last dimension.
    """
    norm = torch.sqrt(torch.sum(tensor**2, dim=1, keepdim=True) + eps)
    return tensor / norm

def torch_squeeze( tensor, min_val, max_val):
    """
    Squeeze the tensor values to the range [min_val, max_val].
    """
    return torch.clamp(tensor, min_val, max_val)

def torch_dot_product(a, b):
    """
    Compute the dot product between two tensors.
    """
    return torch.sum(a * b, dim=-1, keepdim=True)

def preprocess(x):
    """
    preprocess image tensor from [0,1] to [-1,1]
    """
    return x * 2 - 1

def deprocess(x):
    """
    deprocess image tensor from [-1,1] to [0,1]
    """
    return (x + 1) / 2.0

def unpack_svbrdf(svbrdf, is_encoded = False):
    """
    Unpack the SVBRDF tensor into its individual components.
    """
    svbrdf_parts = svbrdf.split(1, dim=-3)

    normals   = None
    diffuse   = None
    roughness = None
    specular  = None
    if not is_encoded:
        normals   = torch.cat(svbrdf_parts[0:3 ], dim=-3)
        diffuse   = torch.cat(svbrdf_parts[3:6 ], dim=-3)
        roughness = torch.cat(svbrdf_parts[6:9 ], dim=-3)
        specular  = torch.cat(svbrdf_parts[9:12], dim=-3)
    else:
        normals   = torch.cat(svbrdf_parts[0:2], dim=-3)
        diffuse   = torch.cat(svbrdf_parts[2:5], dim=-3)
        roughness = torch.cat(svbrdf_parts[5:6], dim=-3)
        specular  = torch.cat(svbrdf_parts[6:9], dim=-3)

    return normals, diffuse, roughness, specular

def pack_svbrdf(normals, diffuse, roughness, specular):
    """
    Pack the individual SVBRDF components into a single tensor.
    """
    return torch.cat([normals, diffuse, roughness, specular], dim=-3)


def create_surface_array(crop_size):
    # Create linearly spaced values for X and Y
    x_surface_array = torch.linspace(-1.0, 1.0, steps=crop_size).unsqueeze(-1)  # Shape: [crop_size, 1]
    x_surface_array = x_surface_array.repeat(1, crop_size)  # Tile across the second dimension
    
    y_surface_array = -1 * x_surface_array.t()  # Transpose and multiply by -1
    
    # Add the last dimension for X and Y
    x_surface_array = x_surface_array.unsqueeze(-1)  # Shape: [crop_size, crop_size, 1]
    y_surface_array = y_surface_array.unsqueeze(-1)  # Shape: [crop_size, crop_size, 1]
    
    # Create a zero array for Z values
    z_surface_array = torch.zeros(crop_size, crop_size, 1)  # Shape: [crop_size, crop_size, 1]
    
    # Concatenate X, Y, and Z to form the surface array
    surface_array = torch.cat([x_surface_array, y_surface_array, z_surface_array], dim=-1)  # Shape: [crop_size, crop_size, 3]
    
    # Add a batch dimension
    surface_array = surface_array.unsqueeze(0)  # Shape: [1, crop_size, crop_size, 3]
    
    return surface_array