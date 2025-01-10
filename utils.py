import torch

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

def gamma_encode(images):
    """
    Gamma encode images
    """
    return torch.pow(images, 1.0/2.2)