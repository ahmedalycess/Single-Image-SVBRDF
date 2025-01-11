import torch
import random
import numpy as np
import math

def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute the dot product of two tensors along the channel dimension. 
        Args:
            a: First tensor
            b: Second tensor
        Returns:
            Tensor containing the dot product along the channel dimension
    """
    return torch.sum(a * b, dim=-3, keepdim=True)

def normalize(a: torch.Tensor) -> torch.Tensor:
    """
    Normalize a tensor along the channel dimension.
        Args:
            a: Tensor to normalize
        Returns:
            Normalized tensor
    """
    return torch.nn.functional.normalize(a, p=2, dim=-3)

def heaviside_step(x: torch.Tensor) -> torch.Tensor:
    """
    Compute the Heaviside step function for a tensor.
        Args:
            x: Tensor to compute the Heaviside step function for
        Returns:
            Tensor containing the Heaviside step function for the input tensor
    """
    return (x > 0.0).float()

def deprocess(tensor):
    """
    Transforms a tensor from range [-1, 1] to [0, 1]
    """
    return (tensor + 1) / 2

def preprocess(tensor):
    """
    Transforms a tensor from range [0, 1] to [-1, 1]
    """
    return tensor * 2 - 1

def pack_svbrdf(normals, diffuse, roughness, specular):
    """
    Pack the SVBRDF into a single tensor.
        Args:
            normals:   Tensor containing the normals
            diffuse:   Tensor containing the diffuse albedo
            roughness: Tensor containing the roughness
            specular:  Tensor containing the specular albedo
        Returns:
            Packed SVBRDF tensor
    """
    return torch.cat([normals, diffuse, roughness, specular], dim=-3)

def unpack_svbrdf(svbrdf, is_encoded = False):
    """
    Unpack the SVBRDF tensor into its components.
        Args:
            svbrdf: Packed SVBRDF tensor
        Returns:
            Tuple containing the normals, diffuse albedo, roughness, and specular albedo
    """
    svbrdf_parts = svbrdf.split(1, dim=-3) # Split the tensor along the channel dimension

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

def decode_svbrdf(svbrdf):
    """
    Decode the SVBRDF tensor from [B, 9, H, W] to [B. 12, H, W]
        Args:
            svbrdf: SVBRDF tensor
        Returns:
            Decoded SVBRDF tensor
    """
    normals_xy, diffuse, roughness, specular  = unpack_svbrdf(svbrdf, True)

    # Repeat roughness channel three times
    # The weird syntax is due to uniform handling of batches of SVBRDFs and single SVBRDFs
    roughness_repetition     = [1] * len(diffuse.shape)
    roughness_repetition[-3] = 3
    roughness = roughness.repeat(roughness_repetition)

    normals_x, normals_y = torch.split(normals_xy.mul(3.0), 1, dim=-3)
    normals_z            = torch.ones_like(normals_x)
    normals              = torch.cat([normals_x, normals_y, normals_z], dim=-3)
    norm                 = torch.sqrt(torch.sum(torch.pow(normals, 2.0), dim=-3, keepdim=True))
    normals              = torch.div(normals, norm)

    return pack_svbrdf(normals, diffuse, roughness, specular)

def generate_normalized_random_direction(count, min_eps = 0.001, max_eps = 0.05):
    """
    Generate a random direction vector with uniform distribution on the unit sphere
        Args:
            count: Number of random directions to generate
            min_eps: Lower bound for the random values
            max_eps: Upper bound for the random values
        Returns:
            Tensor containing the random directions
    """
    r1 = torch.Tensor(count, 1).uniform_(0.0 + min_eps, 1.0 - max_eps)
    r2 = torch.Tensor(count, 1).uniform_(0.0, 1.0)

    r   = torch.sqrt(r1)
    phi = 2 * math.pi * r2
        
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - r**2)

    return torch.cat([x, y, z], axis=-1)

def gamma_encode(images):
    """
    Gamma encode images
    """
    return torch.pow(images, 1.0/2.2)

def enable_deterministic_random_engine(seed=313):
    """
    Enable deterministic random engine for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    torch.manual_seed(seed)