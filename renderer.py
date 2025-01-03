import torch
import math
from utils import torch_normalize, torch_squeeze, torch_dot_product, deprocess

class Renderer:
    """
    Renderer class for rendering SVBRDFs.
    """
    def __init__(self):
        pass

    def render(self, svbrdf, wi, wo, include_diffuse=True):
        """
        Render the SVBRDF using a local shading model

        Parameters
        ---------------------
        svbrdf : torch.Tensor
            The SVBRDF to render. Shape: (BatchSize, Width, Height, 4 * 3)
        wi : torch.Tensor
            The incoming light direction. Shape: (BatchSize, 1, 1, 3)
        wo : torch.Tensor
            The outgoing light direction. Shape: (BatchSize, 1, 1, 3)
        include_diffuse : bool

        Returns
        ---------------------
        torch.Tensor
            The rendered image. Shape: (BatchSize, Width, Height, 3)
        """
        
        wiNorm = torch_normalize(wi).to(svbrdf.device)
        woNorm = torch_normalize(wo).to(svbrdf.device)
        h = torch_normalize((wiNorm + woNorm) / 2.0).to(svbrdf.device)
        diffuse = torch_squeeze(deprocess(svbrdf[:, :, :, 3:6]), 0.0, 1.0).to(svbrdf.device)
        normals = svbrdf[:, :, :, 0:3]
        specular = torch_squeeze(deprocess(svbrdf[:, :, :, 9:12]), 0.0, 1.0).to(svbrdf.device)
        roughness = torch_squeeze(deprocess(svbrdf[:, :, :, 6:9]), 0.0, 1.0).to(svbrdf.device)
        roughness = torch.clamp(roughness, min=0.001).to(svbrdf.device)

        NdotH = torch_dot_product(normals, h)
        NdotL = torch_dot_product(normals, wiNorm)
        NdotV = torch_dot_product(normals, woNorm)
        VdotH = torch_dot_product(woNorm, h)

        diffuse_rendered = self._render_diffuse_Substance(diffuse, specular)
        D_rendered = self._render_D_GGX_Substance(roughness, torch.clamp(NdotH, min=0.0))
        G_rendered = self._render_G_GGX_Substance(roughness, torch.clamp(NdotL, min=0.0), torch.clamp(NdotV, min=0.0))
        F_rendered = self._render_F_GGX_Substance(specular, torch.clamp(VdotH, min=0.0))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered

        if include_diffuse:
            result += diffuse_rendered

        lampIntensity = 1.0
        lampFactor = lampIntensity * math.pi

        result *= lampFactor
        result *= torch.clamp(NdotL, min=0.0) / torch.clamp(wiNorm[..., 2:3], min=0.001)

        return [result, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]

    def _render_diffuse_Substance(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi

    def _render_D_GGX_Substance(self, roughness, NdotH):
        alpha = roughness ** 2
        underD = 1 / torch.clamp((NdotH ** 2 * (alpha ** 2 - 1.0) + 1.0), min=0.001)
        return ((alpha * underD) ** 2) / math.pi

    def _render_F_GGX_Substance(self, specular, VdotH):
        sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
        return specular + (1.0 - specular) * sphg

    def _render_G_GGX_Substance(self, roughness, NdotL, NdotV):
        return self._G1_Substance(NdotL, (roughness ** 2) / 2) * self._G1_Substance(NdotV, (roughness ** 2) / 2)

    def _G1_Substance(self, NdotW, k):
        return 1.0 / torch.clamp((NdotW * (1.0 - k) + k), min=0.001)