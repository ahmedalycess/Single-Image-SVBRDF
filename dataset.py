import matplotlib.pyplot as plt
import os
import torch
import utils

class MaterialDataset(torch.utils.data.Dataset):
    """
    Dataset class for the SVBRDF dataset.
    """

    def __init__(self, data_directory: str, image_size: int):

        self.data_directory = data_directory
        self.file_paths     = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]
        self.image_size     = image_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        input_image, svbrdf = self.read_sample(self.file_paths[idx])
        
        return {'input': input_image, 'svbrdf': svbrdf}

    def read_sample(self, file_path):
        
        full_image = torch.Tensor(plt.imread(file_path)).permute(2, 0, 1)
        
        image_parts = torch.cat(full_image.unsqueeze(0).chunk(chunks= 5, dim=-1), 0) # [5, 3, 256, 256]
        
        input_image = image_parts[0] # [3, 256, 256]
        
        normals     = image_parts[1].unsqueeze(0)
        normals     = utils.preprocess(normals)
        diffuse     = image_parts[2].unsqueeze(0)
        roughness   = image_parts[3].unsqueeze(0)
        specular    = image_parts[4].unsqueeze(0)

        svbrdf = utils.pack_svbrdf(normals, diffuse, roughness, specular).squeeze(0) # [12, 256, 256]

        return input_image, svbrdf