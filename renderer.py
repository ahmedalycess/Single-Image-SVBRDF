import torch
import math
import utils
class Renderer:
    """
    Renderer class for rendering SVBRDFs.
    """
    def render(self, svbrdf, wi, wo, include_diffuse=True):
        """
        Render the SVBRDF using a local shading model.

        Parameters
        ---------------------
        svbrdf : torch.Tensor
            The SVBRDF to render. Shape: (BatchSize, 12, 256, 256)
        wi : torch.Tensor
            The incoming light direction. Shape: (BatchSize, 3, 1, 1)
        wo : torch.Tensor
            The outgoing light direction. Shape: (BatchSize, 3, 1, 1)
        include_diffuse : bool

        Returns
        ---------------------
        torch.Tensor
            The rendered image. Shape: (BatchSize, 3, 256, 256)
        """
        
        wiNorm = utils.torch_normalize(wi).to(svbrdf.device) # (BatchSize, 3, 1, 1)
        woNorm = utils.torch_normalize(wo).to(svbrdf.device) # (BatchSize, 3, 1, 1)
        h      = utils.torch_normalize((wiNorm + woNorm) / 2.0).to(svbrdf.device) # (BatchSize, 3, 1, 1)


        normals   = svbrdf[:, 0:3 , :, :]   # (BatchSize, 3, 256, 256)
        diffuse   = svbrdf[:, 3:6 , :, :]   # (BatchSize, 3, 256, 256)
        roughness = svbrdf[:, 6:9 , :, :]   # (BatchSize, 3, 256, 256)
        specular  = svbrdf[:, 9:12, :, :]   # (BatchSize, 3, 256, 256)
        
        # # print min and max values of normals, diffuse, roughness, specular
        # print("min and max of normals: ", torch.min(normals), torch.max(normals))
        # print("min and max of diffuse: ", torch.min(diffuse), torch.max(diffuse))
        # print("min and max of roughness: ", torch.min(roughness), torch.max(roughness))
        # print("min and max of specular: ", torch.min(specular), torch.max(specular))
        
        # Adjusting the range of the maps to render
        if torch.min(normals) >= 0.0 and torch.max(normals) <= 1.0:
            normals  = utils.preprocess(normals)
        if torch.min(diffuse) < 0.0 or torch.min(roughness) < 0.0 or torch.min(specular) < 0.0:
            diffuse   = utils.deprocess(diffuse  )
            roughness = utils.deprocess(roughness)
            specular  = utils.deprocess(specular )
    
        diffuse   = torch.clamp(diffuse,   0.0,   1.0).to(svbrdf.device) # clamp diffuse to   [0,     1.0]
        roughness = torch.clamp(roughness, 0.001, 1.0).to(svbrdf.device) # clamp roughness to [0.001, 1.0]
        specular  = torch.clamp(specular,  0.0,   1.0).to(svbrdf.device) # clamp specular to  [0,     1.0]

        # # print min and max values of diffuse, roughness, specular
        # print("AFTER: min and max of diffuse: ", torch.min(diffuse), torch.max(diffuse))
        # print("AFTER: min and max of roughness: ", torch.min(roughness), torch.max(roughness))
        # print("AFTER: min and max of specular: ", torch.min(specular), torch.max(specular))
        
        NdotH = utils.torch_dot_product(normals, h     )
        NdotL = utils.torch_dot_product(normals, wiNorm)
        NdotV = utils.torch_dot_product(normals, woNorm)
        VdotH = utils.torch_dot_product(woNorm,  h     )

        # # print min and max values of NdotH, NdotL, NdotV, VdotH
        # print("min and max of NdotH: ", torch.min(NdotH), torch.max(NdotH))
        # print("min and max of NdotL: ", torch.min(NdotL), torch.max(NdotL))
        # print("min and max of NdotV: ", torch.min(NdotV), torch.max(NdotV))
        # print("min and max of VdotH: ", torch.min(VdotH), torch.max(VdotH))

        diffuse_rendered = self._render_diffuse_Substance(diffuse, specular)
        D_rendered       = self._render_D_GGX_Substance(roughness, torch.clamp(NdotH, min=0.0))
        G_rendered       = self._render_G_GGX_Substance(roughness, torch.clamp(NdotL, min=0.0), torch.clamp(NdotV, min=0.0))
        F_rendered       = self._render_F_GGX_Substance(specular,  torch.clamp(VdotH, min=0.0))
        
        # # print min and max values of diffuse_rendered, D_rendered, G_rendered, F_rendered
        # print("min and max of diffuse_rendered: ", torch.min(diffuse_rendered), torch.max(diffuse_rendered))
        # print("min and max of D_rendered: ", torch.min(D_rendered), torch.max(D_rendered))
        # print("min and max of G_rendered: ", torch.min(G_rendered), torch.max(G_rendered))
        # print("min and max of F_rendered: ", torch.min(F_rendered), torch.max(F_rendered))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered

        if include_diffuse:
            result += diffuse_rendered

        lampIntensity = 1.0
        lampFactor = lampIntensity * math.pi

        result *= lampFactor
        result *= torch.clamp(NdotL, min=0.0) / torch.clamp(wiNorm[:, 2:3, :, :], min=0.001)

        # # print min and max values of specualr_rendered, result
        # print("min and max of specular_rendered: ", torch.min(specular_rendered), torch.max(specular_rendered))
        # print("min and max of result: ", torch.min(result), torch.max(result))

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
    


def main():
    
    from dataset import MaterialDataset
    import matplotlib.pyplot as plt

    # load the dataset
    data_directory = "./DummyData"
    image_size = 256
    dataset = MaterialDataset(data_directory, image_size)
    image_idx = 69
    sample = dataset.__getitem__(image_idx)
    dataset.visualize(image_idx)
    input_image = sample['input']
    svbrdf = sample['svbrdf']
    normals, diffuse, roughness, specular = utils.unpack_svbrdf(svbrdf)
    

    # log the input image, diffuse, and specular maps
    input_image = utils.log_tensor(input_image)
    diffuse     = utils.log_tensor(diffuse)
    specular    = utils.log_tensor(specular)

    # testing the renderer
    # 1. Diffuse only: Set specular=0.0 and roughness=1.0 
    # 2. Specular only: Set diffuse=0.0 and roughness=0.5 
    
    # # 1. Diffuse only
    # specular = torch.zeros_like(specular)
    # roughness = torch.ones_like(roughness)

    # # 2. Specular only
    # diffuse = torch.zeros_like(diffuse)
    # roughness = torch.ones_like(roughness) * 0.5

    svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular)
    input_image = input_image.unsqueeze(0) # add batch dimension
    svbrdf = svbrdf.unsqueeze(0)
    results = []

    for i in range(8):
        # create wi and wo tensors of shape (1, 3, 1, 1)
        current_light_pos = utils.generate_normalized_random_direction(1)   # Shape: [batch_size, 3]
        current_view_pos = utils.generate_normalized_random_direction(1)    # Shape: [batch_size, 3]
        wi = current_light_pos.unsqueeze(2).unsqueeze(3)                    # Shape: [batch_size, 3, 1, 1]
        wo = current_view_pos.unsqueeze(2).unsqueeze(3)                     # Shape: [batch_size, 3, 1, 1]
        
        #print("wi: ", wi)
        #print("wo: ", wo)
        #wi = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
        #wo = torch.tensor([[[[0.0]], [[0.0]], [[1.0]]]])
        
        renderer = Renderer()
        result = renderer.render(svbrdf, wi, wo, include_diffuse=True)

        # normalize all results to [0, 1] 
        for i in range(len(result)):
            result[i] = torch.clamp(result[i], min=0.0, max=1.0)
        
        results.append(result[0])
    
    vis2d = []
    vis2d.append(input_image)
    vis2d.extend(results)
    # make vis2d to be a list of 2D tensors 3x3 instead of current 1D of 9 tensors
    vis2d = torch.stack(vis2d)
    
    _, axs = plt.subplots(3, 3)
    # no axis for all subplots
    for i in range(3):
        for j in range(3):
            axs[i, j].axis('off')
    
    # plot all results list
    for i in range(3):
        for j in range(3):
            if i == 0 and j == 0:
                # set the title input image
                axs[i, j].set_title("Input Image")
            else:
                # set the title of the rendered images
                axs[i, j].set_title("Rendered Image " + str(i*3+j))
            axs[i, j].imshow(vis2d[i*3+j].squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    plt.show()
    

main()