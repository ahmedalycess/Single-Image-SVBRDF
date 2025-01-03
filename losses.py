import torch
import torch.nn as nn
import torch.nn.functional as F
from renderer import Renderer
from helpers import unpack_svbrdf, create_surface_array
from scene import generate_diffuse_rendering, generate_specular_rendering


class SVBRDFL1Loss(nn.Module):
    def forward(self, output, target):
        # Split the SVBRDF into its individual components
        output_normals,  output_diffuse,  output_roughness,  output_specular  = unpack_svbrdf(output)
        target_normals, target_diffuse, target_roughness, target_specular = unpack_svbrdf(target)

        epsilon_l1      = 0.01
        output_diffuse   = torch.log(output_diffuse   + epsilon_l1)
        output_specular  = torch.log(output_specular  + epsilon_l1)
        target_diffuse  = torch.log(target_diffuse  + epsilon_l1)
        target_specular = torch.log(target_specular + epsilon_l1)

        # Compute L1 loss for each component
        loss = (
            nn.functional.l1_loss(output_normals, target_normals)
            + nn.functional.l1_loss(output_diffuse, target_diffuse)
            + nn.functional.l1_loss(output_roughness, target_roughness)
            + nn.functional.l1_loss(output_specular, target_specular)
        )

        return loss
    
class SVBRDFL2Loss(nn.Module):
    def forward(self, output, target):
        # Split the SVBRDF into its individual components
        output_normals, output_diffuse, output_roughness, output_specular = unpack_svbrdf(output)
        target_normals, target_diffuse, target_roughness, target_specular = unpack_svbrdf(target)

        epsilon_l2 = 0.01
        output_diffuse = torch.log(output_diffuse + epsilon_l2)
        output_specular = torch.log(output_specular + epsilon_l2)
        target_diffuse = torch.log(target_diffuse + epsilon_l2)
        target_specular = torch.log(target_specular + epsilon_l2)

        # Compute L2 loss for each component
        loss = (
            nn.functional.mse_loss(output_normals, target_normals)
            + nn.functional.mse_loss(output_diffuse, target_diffuse)
            + nn.functional.mse_loss(output_roughness, target_roughness)
            + nn.functional.mse_loss(output_specular, target_specular)
        )

        return loss

class RenderingLoss(nn.Module):
    def __init__(self, nb_diffuse_rendering = 3, nb_specular_rendering = 6, loss_type = "render"):
        super(RenderingLoss, self).__init__()
        self.renderer = Renderer()
        self.nb_diffuse_rendering   = nb_diffuse_rendering
        self.nb_specular_rendering = nb_specular_rendering
        self.loss_type = loss_type

    def forward(self,output, target):
        batch_size = output.shape[0]
        rendered_diffuse_images_outputs = []
        rendered_diffuse_images_targets = []
    
        for _ in range(self.nb_diffuse_rendering):
            diffuse_renderings = generate_diffuse_rendering(batch_size, target, output, self.renderer.render)
            rendered_diffuse_images_targets.append(diffuse_renderings[0][0])
            rendered_diffuse_images_outputs.append(diffuse_renderings[1][0])

        rendered_specular_images_targets = []
        rendered_specular_images_outputs = []
    
        for _ in range(self.nb_specular_rendering):
            specular_renderings = generate_specular_rendering(batch_size=batch_size, surface_array=create_surface_array(crop_size=1), targets=target, outputs=output, render_fn=self.renderer.render, include_diffuse=True)
            rendered_specular_images_targets.append(specular_renderings[0][0])
            rendered_specular_images_outputs.append(specular_renderings[1][0])


        rerendered_targets = torch.cat(rendered_diffuse_images_targets + rendered_specular_images_targets, dim=-1)
        rerendered_outputs = torch.cat(rendered_diffuse_images_outputs + rendered_specular_images_outputs, dim=-1)

        if self.loss_type == "render":
            gen_loss = F.l1_loss(torch.log(rerendered_targets + 0.01), torch.log(rerendered_outputs + 0.01))
        elif self.loss_type == "renderL2":
            gen_loss = F.mse_loss(torch.log(rerendered_targets + 0.01), torch.log(rerendered_outputs + 0.01))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
    
        return gen_loss



class MixedLoss(nn.Module):
    def __init__(self, l1_weight = 0.1, l2_weight = 0.05):
        super(MixedLoss, self).__init__()

        # l1_weight scales the contribution of the SVBRDFL1Loss to the total loss
        self.l1_weight = l1_weight
         # l2_weight scales the contribution of the SVBRDFL1Loss to the total loss
        self.l2_weight = l2_weight
        self.l1_loss = SVBRDFL1Loss()
        self.l2_loss = SVBRDFL2Loss()
        self.rendering_loss = RenderingLoss()

    def forward(self, output, target):
        l1 = self.l1_loss.forward(output, target)
        l2 = self.l2_loss.forward(output, target)
        rendering = self.rendering_loss.forward(output, target)

        return self.l1_weight * l1 + self.l2_weight * l2 + rendering


if __name__ == "__main__":
    # Test the losses
    batch_size = 4
    output = torch.rand(batch_size, 12, 256, 256)
    target = torch.rand(batch_size, 12, 256, 256)


    
    l1_loss = SVBRDFL1Loss()
    loss = l1_loss.forward(output, target)
    print(loss)

    l2_loss = SVBRDFL2Loss()
    loss = l2_loss.forward(output, target)
    print(loss)

    rendering_loss = RenderingLoss()
    loss = rendering_loss.forward(output, target)
    print(loss)

    mixed_loss = MixedLoss()
    loss = mixed_loss.forward(output, target)
    print(loss)