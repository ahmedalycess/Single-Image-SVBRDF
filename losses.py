import torch
import torch.nn as nn
import renderers

def unpack_svbrdf(svbrdf, is_encoded = False):
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

class SVBRDFL1Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        input_normals,  input_diffuse,  input_roughness,  input_specular  = self.unpack_svbrdf(input)
        target_normals, target_diffuse, target_roughness, target_specular = self.unpack_svbrdf(target)

        epsilon_l1      = 0.01
        input_diffuse   = torch.log(input_diffuse   + epsilon_l1)
        input_specular  = torch.log(input_specular  + epsilon_l1)
        target_diffuse  = torch.log(target_diffuse  + epsilon_l1)
        target_specular = torch.log(target_specular + epsilon_l1)

        # Compute L1 loss for each component
        loss = (
            nn.functional.l1_loss(input_normals, target_normals)
            + nn.functional.l1_loss(input_diffuse, target_diffuse)
            + nn.functional.l1_loss(input_roughness, target_roughness)
            + nn.functional.l1_loss(input_specular, target_specular)
        )

        return loss
    
class SVBRDFL2Loss(nn.Module):
    def forward(self, input, target):
        # Split the SVBRDF into its individual components
        input_normals, input_diffuse, input_roughness, input_specular = unpack_svbrdf(input)
        target_normals, target_diffuse, target_roughness, target_specular = unpack_svbrdf(target)

        epsilon_l2 = 0.01
        input_diffuse = torch.log(input_diffuse + epsilon_l2)
        input_specular = torch.log(input_specular + epsilon_l2)
        target_diffuse = torch.log(target_diffuse + epsilon_l2)
        target_specular = torch.log(target_specular + epsilon_l2)

        # Compute L2 loss for each component
        loss = (
            nn.functional.mse_loss(input_normals, target_normals)
            + nn.functional.mse_loss(input_diffuse, target_diffuse)
            + nn.functional.mse_loss(input_roughness, target_roughness)
            + nn.functional.mse_loss(input_specular, target_specular)
        )

        return loss

class RenderingLoss(nn.Module):
    def __init__(self, renderer):
        super(RenderingLoss, self).__init__()
        
        self.renderer = renderer
        self.random_configuration_count   = 3
        self.specular_configuration_count = 6

    def forward(self, input, target):
        batch_input_renderings = self.renderer.render_batch(input)  # Renderer has to support Batch-Rendering
        batch_target_renderings = self.renderer.render_batch(target)

        #logarithmic transformation: applied to the rendered images with a small epsilon for stability
        epsilon_render    = 0.1
        batch_input_renderings_logged  = torch.log(torch.stack(batch_input_renderings, dim=0)  + epsilon_render)
        batch_target_renderings_logged = torch.log(torch.stack(batch_target_renderings, dim=0) + epsilon_render)

        # uses L1 loss on the logarithmic space of the rendered images
        loss = nn.functional.l1_loss(batch_input_renderings_logged, batch_target_renderings_logged)

        return loss

class MixedLoss(nn.Module):
    def __init__(self, renderer, l1_weight = 0.1, l2_weight = 0.05):
        super(MixedLoss, self).__init__()

        # l1_weight scales the contribution of the SVBRDFL1Loss to the total loss
        self.l1_weight = l1_weight
         # l2_weight scales the contribution of the SVBRDFL1Loss to the total loss
        self.l2_weight = l2_weight
        self.l1_loss = SVBRDFL1Loss()
        self.l2_loss = SVBRDFL2Loss
        self.rendering_loss = RenderingLoss(renderer)

    def forward(self, input, target):
        l1 = self.l1_loss(input, target)
        l2 = self.l2_loss(input, target)
        rendering = self.rendering_loss(input, target)

        return self.l1_weight * l1 + self.l2_weight * l2 + rendering
        # TODO define if cases which losses to combine when (dependent on set flags?)