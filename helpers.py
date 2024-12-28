import torch
import torch.nn as nn
import torch.optim as optim
import math

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