from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from helpers import preprocess, deprocess

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

        full_image   = torch.Tensor(plt.imread(self.file_paths[idx])).permute(2, 0, 1)
        # resize the image to width self.image_size* 5 and height self.image_size
        full_image = torch.nn.functional.interpolate(full_image.unsqueeze(0), size=(self.image_size, self.image_size*5), mode='bilinear').squeeze(0) 
        image_parts = torch.cat(full_image.unsqueeze(0).chunk(5, dim=-1), 0) # [5 , 3, 256, 256]
        #print(" image_parts.shape", image_parts.shape)
        # Read the SVBRDF
        svbrdf = None
        normals   = preprocess(image_parts[1].unsqueeze(0))
        diffuse   = preprocess(image_parts[2].unsqueeze(0))
        roughness = preprocess(image_parts[3].unsqueeze(0))
        specular  = preprocess(image_parts[4].unsqueeze(0))
        svbrdf = torch.cat([normals, diffuse, roughness, specular], dim=-3).squeeze(0) # [12, 256, 256]
        #print("svbrdf.shape: ", svbrdf.shape)
        input_image = preprocess(image_parts[0]) # [3, 256, 256]
        
        print("input_image.shape: ", input_image.shape)

        return {'input': input_image, 'svbrdf': svbrdf}

    # get length of the dataset
    def __len__(self):
        return len(self.file_paths)

    # visualize the dataset item using matplotlib
    def visualize(self, idx):
        # deprocess images before visualization
        sample = self.__getitem__(idx)
        input_image = deprocess(sample['input']).permute(1, 2, 0)
        #print("input_image.shape: ", input_image.shape)
        svbrdf = sample['svbrdf']
        svbrdf_parts = svbrdf.split(1, dim=-3)
        normals   = torch.cat(svbrdf_parts[0:3 ], dim=-3)
        diffuse   = torch.cat(svbrdf_parts[3:6 ], dim=-3)
        roughness = torch.cat(svbrdf_parts[6:9 ], dim=-3)
        specular  = torch.cat(svbrdf_parts[9:12], dim=-3)
        
        # transpose the image to be in the format (height, width, channels)
        normals = deprocess(normals).permute(1, 2, 0)
        diffuse = deprocess(diffuse).permute(1, 2, 0)
        roughness = deprocess(roughness).permute(1, 2, 0)
        specular = deprocess(specular).permute(1, 2, 0)

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
    dataset.visualize(75)

if __name__ == '__main__':
    main()