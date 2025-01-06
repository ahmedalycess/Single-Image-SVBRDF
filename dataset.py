from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from utils import unpack_svbrdf, pack_svbrdf

class MaterialDataset(torch.utils.data.Dataset):
    """
    Class representing a collection of SVBRDF samples with corresponding input images (rendered or real views of the SVBRDF)
    """

    def __init__(self, data_directory, image_size):
        self.data_directory = data_directory #./DummyData
        self.file_paths = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f))]

        self.image_size = image_size #256

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        full_image  = torch.Tensor(plt.imread(self.file_paths[idx])).permute(2, 0, 1) # [channels, height, width] --> [3, 288, 1440] 

        # resize the image to width self.image_size* 5 and height self.image_size
        full_image = torch.nn.functional.interpolate(full_image.unsqueeze(0), size=(self.image_size, self.image_size*5), mode='bilinear').squeeze(0) 
        image_parts = torch.cat(full_image.unsqueeze(0).chunk(5, dim=-1), 0) # [5 , 3, 256, 256]

        input_image = image_parts[0] # [3, 256, 256]
        normals     = image_parts[1] # [3, 256, 256]
        diffuse     = image_parts[2] # [3, 256, 256]
        roughness   = image_parts[3] # [3, 256, 256]
        specular    = image_parts[4] # [3, 256, 256]
        
        svbrdf = pack_svbrdf(normals=normals, diffuse=diffuse, roughness=roughness, specular=specular) # [12, 256, 256]  

        return {'input': input_image, 'svbrdf': svbrdf}

    # get length of the dataset
    def __len__(self):
        return len(self.file_paths)

    # visualize the dataset item using matplotlib
    def visualize(self, idx):
        # deprocess images before visualization
        sample = self.__getitem__(idx)
    
        input_image = sample['input']
        svbrdf = sample['svbrdf']

        normals, diffuse, roughness, specular = unpack_svbrdf(svbrdf)
        

        input_image = input_image.permute(1, 2, 0)
        normals     = normals.permute(1, 2, 0)
        diffuse     = diffuse.permute(1, 2, 0)
        roughness   = roughness.permute(1, 2, 0)
        specular    = specular.permute(1, 2, 0)
        
        # Create the figure and axes grid
        fig = plt.figure(figsize=(12, 6))

        # Image 1 on the left
        ax1 = fig.add_axes([0.05, 0.3, 0.25, 0.4])
        ax1.imshow(input_image)
        ax1.axis('off')

        # Images 2 and 3 vertical to the right of Image 1
        ax2 = fig.add_axes([0.35, 0.55, 0.25, 0.4])  # Top-right
        ax2.imshow(normals)
        ax2.axis('off')

        ax3 = fig.add_axes([0.35, 0.05, 0.25, 0.4])  # Bottom-right
        ax3.imshow(diffuse)
        ax3.axis('off')

        # Images 4 and 5 stacked to the far right
        ax4 = fig.add_axes([0.7, 0.55, 0.25, 0.4])  # Top-far-right
        ax4.imshow(roughness)
        ax4.axis('off')

        ax5 = fig.add_axes([0.7, 0.05, 0.25, 0.4])  # Bottom-far-right
        ax5.imshow(specular)
        ax5.axis('off')

        # Display the layout
        plt.show()

def main():
    dataset = MaterialDataset(data_directory='./DummyData', image_size=256)
    print("Dataset length: ", len(dataset))
    dataset.visualize(61)

if __name__ == '__main__':
    main()