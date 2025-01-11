import scene
import torch
import torch.nn as nn
import utils

class SVBRDFL1Loss(nn.Module):
    def forward(self, output, target):

        output_normals,  output_diffuse,  output_roughness,  output_specular  = utils.unpack_svbrdf(output)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(target)

        epsilon_l1      = 1e-2

        output_diffuse   = torch.log(output_diffuse   + epsilon_l1)
        output_specular  = torch.log(output_specular  + epsilon_l1)

        target_diffuse  = torch.log(target_diffuse  + epsilon_l1)
        target_specular = torch.log(target_specular + epsilon_l1)

        return  nn.functional.l1_loss(output_normals  , target_normals  ) + \
                nn.functional.l1_loss(output_diffuse  , target_diffuse  ) + \
                nn.functional.l1_loss(output_roughness, target_roughness) + \
                nn.functional.l1_loss(output_specular , target_specular )
    

class SVBRDFL2Loss(nn.Module):
    def forward(self, output, target):

        output_normals, output_diffuse, output_roughness, output_specular = utils.unpack_svbrdf(output)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(target)

        epsilon_l2      = 1e-2

        output_diffuse  = torch.log(output_diffuse  + epsilon_l2)
        output_specular = torch.log(output_specular + epsilon_l2)

        target_diffuse  = torch.log(target_diffuse  + epsilon_l2)
        target_specular = torch.log(target_specular + epsilon_l2)

       
        return nn.functional.mse_loss(output_normals  , target_normals  ) + \
               nn.functional.mse_loss(output_diffuse  , target_diffuse  ) + \
               nn.functional.mse_loss(output_roughness, target_roughness) + \
               nn.functional.mse_loss(output_specular , target_specular )


class RenderingLoss(nn.Module):
    def __init__(self, renderer):
        super(RenderingLoss, self).__init__()
        
        self.renderer = renderer
        self.random_configuration_count   = 3
        self.specular_configuration_count = 6

    def forward(self, output, target):
        batch_size = output.shape[0]

        batch_output_renderings = []
        batch_target_renderings = []
        for i in range(batch_size):
            scenes = scene.generate_random_scenes(self.random_configuration_count) + scene.generate_specular_scenes(self.specular_configuration_count)
            
            output_svbrdf = output[i]
            target_svbrdf = target[i]
            output_renderings = []
            target_renderings = []

            for s in scenes:
                output_renderings.append(self.renderer.render(s, output_svbrdf))
                target_renderings.append(self.renderer.render(s, target_svbrdf))
            batch_output_renderings.append(torch.cat(output_renderings, dim=0))
            batch_target_renderings.append(torch.cat(target_renderings, dim=0))

        epsilon_render = 1e-2

        batch_output_renderings_logged = torch.log(torch.stack(batch_output_renderings, dim=0) + epsilon_render)
        batch_target_renderings_logged = torch.log(torch.stack(batch_target_renderings, dim=0) + epsilon_render)

        return nn.functional.l1_loss(batch_output_renderings_logged, batch_target_renderings_logged)

class MixedLoss(nn.Module):
    def __init__(self, renderer, l1_weight = 0.1, l2_weight = 0.1):
        super(MixedLoss, self).__init__()

        self.l1_weight      = l1_weight
        self.l1_loss        = SVBRDFL1Loss()
        self.l2_weight      = l2_weight
        self.l2_loss        = SVBRDFL2Loss()
        self.rendering_loss = RenderingLoss(renderer)

    def forward(self, output, target):
        return  self.l1_weight * self.l1_loss(output, target) +\
                self.l2_weight * self.l2_loss(output, target) +\
                self.rendering_loss(output, target)