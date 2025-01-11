import torch
import math
import utils

class LocalRenderer:
    """
    The local renderer is a simple renderer that computes the rendering equation locally for each pixel.
    It uses a microfacet model to compute the specular reflection and a Lambertian model for the diffuse reflection.

    The rendering equation is given by:
        lambertian_term     f_d  = kd * diffuse / pi
        specular_term       f_s  = F * G * D / (4 * VN * LN)
        rendering           f    = f_d + f_s

    """

    def compute_diffuse_term(self, diffuse: torch.Tensor, ks: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffuse term of the BRDF
            kd = (1 - ks)
            f_d = kd * diffuse / pi
        """
        kd = (1.0 - ks)
        return  kd * diffuse / math.pi

    def compute_specular_term(self, wi: torch.Tensor, wo: torch.Tensor, normals: torch.Tensor, 
                            diffuse: torch.Tensor, roughness: torch.Tensor, specular: torch.Tensor) -> torch.Tensor:
        """
        Compute the specular term of the BRDF
            H = normalize((wi + wo) / 2)
            NH = dot(normals, H)
            VH = dot(wo, H)
            LH = dot(wi, H)
            VN = dot(wo, normals)
            LN = dot(wi, normals)

            F = compute_fresnel(specular, VH)
            G = compute_geometry(roughness, VH, LH, VN, LN)
            D = compute_microfacet_distribution(roughness, NH)
            f_s = F * G * D / (4 * VN * LN)
        """

        H = utils.normalize((wi + wo) / 2.0)


        NH  = torch.clamp(utils.dot_product(normals, H),  min=0.001)
        VH  = torch.clamp(utils.dot_product(wo, H),       min=0.001)
        LH  = torch.clamp(utils.dot_product(wi, H),       min=0.001)
        VN  = torch.clamp(utils.dot_product(wo, normals), min=0.001)
        LN  = torch.clamp(utils.dot_product(wi, normals), min=0.001)

        F = self.compute_fresnel(specular, VH)
        G = self.compute_geometry(roughness, VH, LH, VN, LN)
        D = self.compute_microfacet_distribution(roughness, NH)
        
        return F * G * D / (4.0 * VN * LN), F
    
    def compute_microfacet_distribution(self, roughness: torch.Tensor, NH: torch.Tensor) -> torch.Tensor:
        """
        Compute the microfacet distribution term of the BRDF

            alpha = roughness^2
            D = (alpha^2 * heaviside_step(NH)) / (pi * (NH^2 * (alpha^2 + (1 - NH^2) / NH^2))^2)
        """
        alpha            = roughness**2
        nominator   = alpha**2 * utils.heaviside_step(NH)
        denominator = torch.clamp(NH**2 * (alpha**2 + (1.0 - NH**2) / NH**2), 0.001)
        return nominator / (math.pi * denominator**2)

    def compute_fresnel(self, specular: torch.Tensor, VH: torch.Tensor) -> torch.Tensor:
        """
        Compute the Fresnel term of the BRDF
        
            F = specular + (1 - specular) * (1 - VH)^5
        """
        return specular + (1.0 - specular) * (1.0 - VH)**5

    def compute_g1(self, roughness: torch.Tensor, XH: torch.Tensor, XN: torch.Tensor) -> torch.Tensor:
        """
        Compute the shadowing/masking term of the BRDF
        
            alpha = roughness^2
            G1 = 2 * heaviside_step(XH / XN) / (1 + sqrt(1 + alpha^2 * (1 - XN^2) / XN^2))
        """
        alpha = roughness**2
        
        return 2 * utils.heaviside_step(XH / XN) / (1 + torch.sqrt(1 + alpha**2 * (1.0 - XN**2) / XN**2))

    def compute_geometry(self, roughness: torch.Tensor, VH: torch.Tensor, LH: torch.Tensor, 
                         VN: torch.Tensor, LN: torch.Tensor) -> torch.Tensor:
        """
        Compute the geometry term of the BRDF
        
            G = G1(roughness, VH, VN) * G1(roughness, LH, LN)
        """
        return self.compute_g1(roughness, VH, VN) * self.compute_g1(roughness, LH, LN)

    def evaluate_brdf(self, wi: torch.Tensor, wo: torch.Tensor, normals: torch.Tensor, 
                      diffuse: torch.Tensor, roughness: torch.Tensor, specular: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the BRDF for the given input parameters
        """
        specular_term, ks = self.compute_specular_term(wi, wo, normals, diffuse, roughness, specular)
        diffuse_term      = self.compute_diffuse_term(diffuse, ks)
        return diffuse_term + specular_term

    def render(self, scene, svbrdf: torch.Tensor) -> torch.Tensor:
        """
        Render the scene with the given SVBRDF
        """

        device = svbrdf.device

        # Generate surface coordinates for the material patch
        xcoords_row  = torch.linspace(-1, 1, svbrdf.shape[-1], device=device)
        xcoords      = xcoords_row.unsqueeze(0).expand(svbrdf.shape[-2], svbrdf.shape[-1]).unsqueeze(0)
        ycoords      = torch.transpose(xcoords, dim0=1, dim1=2) * -1 
        coords       = torch.cat((xcoords, ycoords, torch.zeros_like(xcoords)), dim=0)


        light_position          = torch.Tensor(scene.light.position).unsqueeze(-1).unsqueeze(-1).to(device)
        relative_light_position = light_position - coords
        wi                      = utils.normalize(relative_light_position)

        view_position           = torch.Tensor(scene.view.position).unsqueeze(-1).unsqueeze(-1).to(device)
        relative_view_position  = view_position - coords
        wo                      = utils.normalize(relative_view_position)

        normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)
        roughness = torch.clamp(roughness, min=0.001)

        f  = self.evaluate_brdf(wi, wo, normals, diffuse, roughness, specular)
        LN = torch.clamp(utils.dot_product(wi, normals), min=0.0)

        light_color = torch.Tensor(scene.light.color).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
        falloff     = 1.0 / torch.sqrt(utils.dot_product(relative_light_position, relative_light_position))**2
        result    = torch.mul(torch.mul(f, light_color * falloff), LN)

        return result


if __name__ == '__main__':
    # Testing code for the renderer(s)
    import dataset
    import scene
    import matplotlib.pyplot as plt
    import utils

    data   = dataset.MaterialDataset(data_directory="./data/train", image_size=256)
    loader = torch.utils.data.DataLoader(data, batch_size=1, pin_memory=False)

    renderer = LocalRenderer()
    scene    = scene.Scene(scene.View([0.0, -1.0, 2.0]), scene.Light([0.0, 0.0, 2.0], [50.0, 50.0, 50.0]))

    fig       = plt.figure(figsize=(8, 8))
    row_count = 2 * len(data)
    col_count = 5
    for i_row, batch in enumerate(loader):
        batch_inputs = batch["input"]
        batch_svbrdf = batch["svbrdf"]

        batch_inputs.squeeze_(0)

        input  = utils.gamma_encode(batch_inputs)
        svbrdf = batch_svbrdf

        normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)

        fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 1)
        plt.imshow(input.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 2)
        plt.imshow(utils.deprocess(normals.squeeze(0).permute(1, 2, 0)))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 3)
        plt.imshow(diffuse.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 4)
        plt.imshow(roughness.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 5)
        plt.imshow(specular.squeeze(0).permute(1, 2, 0))
        plt.axis('off')

        rendering = renderer.render(scene, utils.pack_svbrdf(normals, diffuse, roughness, specular))    
        rendering = utils.gamma_encode(rendering).squeeze(0).permute(1, 2, 0)
        fig.add_subplot(row_count, col_count, 2 * i_row * col_count + 6)
        plt.imshow(rendering)
        plt.axis('off')
    plt.show()