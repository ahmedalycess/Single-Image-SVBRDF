import os
import tensorflow as tf
import torch
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
from lxml import etree
from random import shuffle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to xml file, folder or image (defined by --imageFormat) containing information images")
parser.add_argument("--mode", required=True, choices=["test", "train", "eval"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=1000, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=1000, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")
parser.add_argument("--test_freq", type=int, default=20000, help="test model every test_freq steps, 0 to disable")


parser.add_argument("--testMode", type=str, default="auto", choices=["auto", "xml", "folder", "image"], help="Which loss to use instead of the L1 loss")
parser.add_argument("--imageFormat", type=str, default="png", choices=["jpg", "png", "jpeg", "JPG", "JPEG", "PNG"], help="Which format have the input files")

# to get tracing working on GPU, LD_LIBRARY_PATH may need to be modified:
# LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/extras/CUPTI/lib64

parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=288, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--nbTargets", type=int, default=4, help="Number of images to output")
parser.add_argument("--depthFactor", type=int, default=1, help="Factor for the capacity of the network")
parser.add_argument("--loss", type=str, default="l1", choices=["l1", "specuRough", "render", "flatMean", "l2", "renderL2"], help="Which loss to use instead of the L1 loss")
parser.add_argument("--useLog", dest="useLog", action="store_true", help="Use the log for input")
parser.set_defaults(useLog=False)
parser.add_argument("--logOutputAlbedos", dest="logOutputAlbedos", action="store_true", help="Log the output albedos ? ?")
parser.set_defaults(logOutputAlbedos=False)
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")

parser.add_argument("--nbDiffuseRendering", type=int, default=3, help="Number of diffuse renderings in the rendering loss")
parser.add_argument("--nbSpecularRendering", type=int, default=6, help="Number of specular renderings in the rendering loss")
parser.add_argument("--lr", type=float, default=0.00002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--includeDiffuse", dest="includeDiffuse", action="store_true", help="Include the diffuse term in the specular renderings of the rendering loss ?")
parser.set_defaults(includeDiffuse=True)
parser.add_argument("--correctGamma", dest="correctGamma", action="store_true", help="correctGamma ? ?")
parser.set_defaults(correctGamma=False)

a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

if a.testMode == "auto":
    if a.input_dir.lower().endswith(".xml"):
        a.testMode = "xml";
    elif os.path.isdir(a.input_dir):
        a.testMode = "folder";
    else:
        a.testMode = "image";

Examples = collections.namedtuple("Examples", "iterator, paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs, gen_loss_L1, gen_grads_and_vars, train, rerendered, gen_loss_L1_exact")

if a.depthFactor == 0:
    a.depthFactor = a.nbTargets

def readInputImage(inputPath):
    return [inputPath]
        
    
def readInputFolder(input_dir, shuffleList):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
        
    pathList = glob.glob(os.path.join(input_dir, "*." + a.imageFormat))
    pathList = sorted(pathList);
    
    if shuffleList:
        shuffle(pathList)
    return pathList
    
def readInputXML(inputPath, shuffleList):
    exampleDict = {}
    pathDict = {}
    tree = etree.parse(inputPath)
    for elem in tree.findall('.//item'):
        imagePath = elem.find('image').text
        if not (imagePath is None) and os.path.exists(imagePath):
            lightPower = elem.find('lightPower').text
            lightXPos = elem.find('lightXPos').text
            lightYPos = elem.find('lightYPos').text
            lightZPos = elem.find('lightZPos').text
            camXPos = elem.find('camXPos').text
            camYPos = elem.find('camYPos').text
            camZPos = elem.find('camZPos').text
            uvscale = elem.find('uvscale').text
            uoffset = elem.find('uoffset').text
            voffset = elem.find('voffset').text
            rotation = elem.find('rotation').text
            identifier = elem.find('identifier').text
            
            substanceName = imagePath.split("/")[-1]
            if(substanceName.split('.')[0].isdigit()):
                substanceName = '%04d' % int(substanceName.split('.')[0])
            substanceNumber = 0
            imageSplitsemi = imagePath.split(";")
            if len(imageSplitsemi) > 1:                    
                substanceName = imageSplitsemi[1]
                substanceNumber = imageSplitsemi[2].split(".")[0]
            #def __init__(self, name, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, path):

            material = inputMaterial(substanceName, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, imagePath)
            idkey = str(substanceNumber) +";"+ identifier.rsplit(";", 1)[0]
            
            if not (substanceName in exampleDict) :
                exampleDict[substanceName] = {idkey : [material]}
                pathDict[imagePath] = material # Add only a path to be queried as for each image we will grab the others that are alike with the other dict

            else:
                if not (idkey in exampleDict[substanceName]):
                    exampleDict[substanceName][idkey] = [material]
                    pathDict[imagePath] = material # Add only a path to be queried as for each image we will grab the others that are alike with the other dict

                else:
                    exampleDict[substanceName][idkey].append(material)
    print("dict length : " + str(len(exampleDict.items())))
    flatPathList = createMaterialTable(exampleDict, shuffleList)
    return flatPathList


class MaterialDataset(Dataset):
    def __init__(self, input_dir, mode, scale_size=288, crop_size=256, should_shuffle=True):
        self.crop_size = crop_size
        self.scale_size = scale_size
        
        # Get file paths based on mode
        print('Testing mode : ' + a.testMode)
        print('Input dir : ' + input_dir)
        if a.testMode == "xml":
            self.file_paths = readInputXML(input_dir, should_shuffle)
        elif a.testMode == "folder":
            self.file_paths = readInputFolder(input_dir, should_shuffle)
        elif a.testMode == "image":
            self.file_paths = readInputImage(input_dir)
            
        if len(self.file_paths) == 0:
            raise Exception("input_dir contains no image files")
            
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize(scale_size),
            transforms.ToTensor(),  # Converts to [0,1] float tensor
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        
        # Load input and target images here
        # This part depends on your specific data format
        input_image = Image.open(path['input'])
        target_images = [Image.open(target_path) for target_path in path['targets']]
        
        # Apply transforms
        input_tensor = self.transform(input_image)
        target_tensors = [self.transform(target) for target in target_images]
        
        # Random crop if scale_size > crop_size
        if self.scale_size > self.crop_size:
            i = torch.randint(0, self.scale_size - self.crop_size, (1,))
            j = torch.randint(0, self.scale_size - self.crop_size, (1,))
            
            input_tensor = input_tensor[:, i:i+self.crop_size, j:j+self.crop_size]
            target_tensors = [t[:, i:i+self.crop_size, j:j+self.crop_size] for t in target_tensors]
        
        # Stack target tensors
        target_tensor = torch.stack(target_tensors)
        
        return {
            'path': path,
            'input': input_tensor,
            'target': target_tensor
        }

def load_examples(input_dir, should_shuffle):
    # Create dataset
    dataset = MaterialDataset(
        input_dir=input_dir,
        mode=a.mode,
        scale_size=a.scale_size,
        crop_size=CROP_SIZE,
        should_shuffle=should_shuffle
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=a.batch_size,
        shuffle=should_shuffle,
        num_workers=1,  # Adjust based on your needs
        drop_last=True  # Drop incomplete batches
    )
    
    # Calculate steps per epoch
    steps_per_epoch = len(dataset) // a.batch_size
    
    return Examples(
        iterator=dataloader,
        paths=None,  # Paths will be provided in each batch
        inputs=None,  # Inputs will be provided in each batch
        targets=None,  # Targets will be provided in each batch
        count=len(dataset),
        steps_per_epoch=steps_per_epoch,
    )

class Generator(nn.Module):
    def __init__(self, ngf=64, depth_factor=1):
        super(Generator, self).__init__()
        self.ngf = ngf
        self.depth_factor = depth_factor
        
        # Global Network (FC layers)
        self.global_net = nn.ModuleList([
            nn.Linear(3, ngf * 2),  # First FC layer (3 is for input mean across H,W)
            nn.SELU()
        ])
        
        # Encoder layers
        self.encoder = nn.ModuleList()
        # encoder_1: [batch, 256, 256, 3] => [batch, 128, 128, ngf]
        self.encoder.append(nn.Conv2d(3, ngf * depth_factor, 4, stride=2, padding=1))
        
        # Encoder specs (channels multiplier)
        encoder_specs = [
            ngf * 2 * depth_factor,  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4 * depth_factor,  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8 * depth_factor,  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8 * depth_factor,  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8 * depth_factor,  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8 * depth_factor,  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ngf * 8 * depth_factor,  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
        
        in_channels = ngf * depth_factor
        for out_channels in encoder_specs:
            self.encoder.append(self._make_encoder_layer(in_channels, out_channels))
            in_channels = out_channels
            
        # Decoder specs (channels, dropout)
        decoder_specs = [
            (ngf * 8 * depth_factor, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8 * depth_factor, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8 * depth_factor, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8 * depth_factor, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4 * depth_factor, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2 * depth_factor, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf * depth_factor, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
        ]
        
        # Decoder layers
        self.decoder = nn.ModuleList()
        for i, (out_channels, dropout) in enumerate(decoder_specs):
            self.decoder.append(self._make_decoder_layer(in_channels * 2 if i > 0 else in_channels, 
                                                       out_channels, 
                                                       dropout))
            in_channels = out_channels
            
        # Final layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, 9, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def _make_encoder_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels)
        )
        
    def _make_decoder_layer(self, in_channels, out_channels, dropout=0.0):
        layers = [
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)
    
    def _global_to_generator(self, global_features, out_channels):
        # Convert global features to proper shape for addition to feature maps
        batch_size = global_features.size(0)
        return global_features.view(batch_size, -1, 1, 1).expand(-1, out_channels, 1, 1)
        
    def forward(self, x):
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Global network input
        input_mean = torch.mean(x, dim=[2, 3])  # Mean across H,W dimensions
        global_features = input_mean
        
        # Global network
        for layer in self.global_net:
            global_features = layer(global_features)
            
        # Encoder
        for i, encoder_layer in enumerate(self.encoder):
            if i == 0:
                x = encoder_layer(x)
            else:
                x = encoder_layer(x)
                # Update global features
                mean = torch.mean(x, dim=[2, 3])
                global_features = torch.cat([global_features.unsqueeze(2).unsqueeze(3), 
                                          mean.unsqueeze(2).unsqueeze(3)], dim=1)
                x = x + self._global_to_generator(global_features, x.size(1))
            encoder_outputs.append(x)
            
        # Decoder with skip connections
        for i, decoder_layer in enumerate(self.decoder):
            if i == 0:
                x = decoder_layer(encoder_outputs[-1])
            else:
                x = torch.cat([x, encoder_outputs[-(i+1)]], dim=1)
                x = decoder_layer(x)
                
        # Final layer with skip connection to first encoder layer
        x = torch.cat([x, encoder_outputs[0]], dim=1)
        x = self.final_layer(x)
        
        return x

def create_model(device='cuda'):
    """
    Create the model, optimizer, and loss functions
    """
    # Initialize generator
    generator = Generator(ngf=a.ngf).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        generator.parameters(),
        lr=a.lr,
        betas=(a.beta1, 0.999)
    )

    def process_output(output):
        """Process the generator output into the required components"""
        batch_size = output.size(0)
        
        # Split output into components [batch, channels, height, width]
        partial_normals = output[:, 0:2]  # First 2 channels
        diffuse = output[:, 2:5]         # Next 3 channels
        roughness = output[:, 5:6]       # 1 channel
        specular = output[:, 6:9]        # Last 3 channels

        # Create ones tensor for normals
        ones = torch.ones(batch_size, 1, CROP_SIZE, CROP_SIZE, device=device)
        
        # Normalize normals
        normals = torch.cat([partial_normals, ones], dim=1)
        normals = F.normalize(normals, dim=1)  # Normalize along channel dimension
        
        # Reconstruct full output
        roughness_expanded = roughness.repeat(1, 3, 1, 1)  # Repeat roughness 3 times
        reconstructed = torch.cat([
            normals,
            diffuse,
            roughness_expanded,
            specular
        ], dim=1)
        
        return reconstructed

    def compute_loss(outputs, targets):
        """Compute the loss based on the specified loss type"""
        if a.loss == "l1":
            epsilon = 0.001
            # Assuming outputs and targets are properly aligned tensors
            normal_loss = torch.abs(targets[:, 0:3] - outputs[:, 0:3]) * a.normalLossFactor
            
            # Log space for diffuse and specular
            diffuse_loss = torch.abs(
                torch.log(epsilon + deprocess_torch(targets[:, 3:6])) - 
                torch.log(epsilon + deprocess_torch(outputs[:, 3:6]))
            ) * a.diffuseLossFactor
            
            roughness_loss = torch.abs(targets[:, 6:9] - outputs[:, 6:9]) * a.roughnessLossFactor
            
            specular_loss = torch.abs(
                torch.log(epsilon + deprocess_torch(targets[:, 9:12])) - 
                torch.log(epsilon + deprocess_torch(outputs[:, 9:12]))
            ) * a.specularLossFactor
            
            total_loss = torch.mean(normal_loss + diffuse_loss + roughness_loss + specular_loss)
            return total_loss
            
        elif a.loss == "l2":
            epsilon = 0.001
            # Similar to L1 but using squared differences
            normal_loss = torch.square(targets[:, 0:3] - outputs[:, 0:3]) * a.normalLossFactor
            # ... similar pattern for other components ...
            return total_loss
            
        # Add other loss types as needed
        
        return None  # Placeholder for other loss types

    def train_step(inputs, targets):
        """Perform a single training step"""
        generator.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = generator(inputs)
        processed_outputs = process_output(outputs)
        
        # Compute loss
        loss = compute_loss(processed_outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return {
            'outputs': processed_outputs,
            'loss': loss.item()
        }

    def eval_step(inputs, targets):
        """Perform a single evaluation step"""
        generator.eval()
        with torch.no_grad():
            outputs = generator(inputs)
            processed_outputs = process_output(outputs)
            loss = compute_loss(processed_outputs, targets)
            
        return {
            'outputs': processed_outputs,
            'loss': loss.item()
        }

    return {
        'generator': generator,
        'optimizer': optimizer,
        'train_step': train_step,
        'eval_step': eval_step,
        'process_output': process_output
    }

# Helper function for deprocessing
def deprocess_torch(x):
    """Convert from [-1,1] to [0,1]"""
    return (x + 1) / 2

def save_loss_value(values):
    averaged = np.mean(values)
    with open(os.path.join(a.output_dir, "losses.txt"), "a") as f:
            f.write(str(averaged) + "\n")
            
def save_images(fetches, output_dir = a.output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        #fetch inputs
        kind = "inputs"
        filename = name + "-" + kind + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][i]
        with open(out_path, "wb") as f:
            f.write(contents)
        #fetch outputs and targets
        for kind in ["outputs", "targets"]:
            for idImage in range(a.nbTargets):
                filename = name + "-" + kind + "-" + str(idImage) + "-.png"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
                filetsetKey = kind + str(idImage)
                fileset[filetsetKey] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i * a.nbTargets + idImage]
                with open(out_path, "wb") as f:
                    f.write(contents)
        filesets.append(fileset)
    return filesets

def append_index(filesets, output_dir = a.output_dir, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        mapnames = ["normals", "diffuse", "roughness", "log(specular)"]
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>log(input)</th>")
        for idImage in range(a.nbTargets):
            index.write("<th>" + str(mapnames[idImage]) + "</th>")
        index.write("</tr>")            

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s targets</td>" % fileset["name"])
        if a.mode != "eval" : 

            for kind in ["inputs", "targets"]:
                if kind == "inputs":
                    index.write("<td><img src='images/%s'></td>" % fileset[kind])
                elif kind == "targets":
                    for idImage in range(a.nbTargets):
                        filetsetKey = kind + str(idImage)
                        index.write("<td><img src='images/%s'></td>" % fileset[filetsetKey])
            index.write("</tr>")
            index.write("<tr>")

        if step:
            index.write("<td></td>")
        index.write("<td>outputs</td>")
        for kind in ["inputs", "outputs"]:
            if kind == "inputs":
                index.write("<td><img src='images/%s'></td>" % fileset[kind])
            elif kind=="outputs":
                for idImage in range(a.nbTargets):
                    filetsetKey = kind + str(idImage)
                    index.write("<td><img src='images/%s'></td>" % fileset[filetsetKey])
        index.write("</tr>")
    
    return index_path

def run_test_from_train(current_step, eval_examples, max_steps, display_fetches_test, model, device):
    """
    Run evaluation from training using PyTorch.

    Args:
        current_step (int): Current training step.
        eval_examples (DataLoader or iterable): Iterable containing evaluation data.
        max_steps (int): Maximum number of evaluation steps.
        ??? is it a dict? display_fetches_test (dict): Predefined structure describing what to fetch (inputs, outputs, etc.).
        model (torch.nn.Module): Trained PyTorch model.
        device (torch.device): Device to run the evaluation on (e.g., "cuda" or "cpu").
    """
    model.eval()  # Set model to evaluation mode
    test_output_dir = os.path.join(a.output_dir, f"testStep{current_step}")
    os.makedirs(test_output_dir, exist_ok=True)
    
    index_path = None
    filesets = []

    # Limit max_steps to the number of available batches
    max_steps = min(len(eval_examples), max_steps)

    with torch.no_grad():  # Disable gradient computation for evaluation
        for step, batch in enumerate(eval_examples):
            if step >= max_steps:
                break

            try:
                # Assuming `batch` contains inputs and targets
                inputs, targets = batch  # Modify if batch structure is different
                inputs, targets = inputs.to(device), targets.to(device)

                # Get model predictions
                outputs = model(inputs)

                # Structure results based on display_fetches_test
                results_test = {
                    "paths": [f"image_{step}_{i}.png" for i in range(len(inputs))],
                    "inputs": [input_img.cpu().numpy().tobytes() for input_img in inputs],
                    "outputs": [output_img.cpu().numpy().tobytes() for output_img in outputs],
                    "targets": [target_img.cpu().numpy().tobytes() for target_img in targets],
                }

                # Save images and update index
                filesets += save_images(results_test, test_output_dir)

            except Exception as e:
                print(f"Error in run_test_from_train: {e}")
                continue

    # Generate index HTML for the saved images
    if filesets:
        index_path = append_index(filesets, test_output_dir)

    print("wrote index at", index_path)

def reshape_tensor_display(tensor, split_amount, log_albedo=False):
    """
    Process a PyTorch tensor by splitting, optionally applying a log operation, stacking, and reshaping.

    Args:
        tensor (torch.Tensor): Input tensor of shape [batch, 256, 256, channels].
        split_amount (int): Number of splits along the last dimension (channels).
        log_albedo (bool): Whether to apply a log operation to certain splits.

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    # Split the tensor along the last dimension (axis=-1 in PyTorch)
    tensors_list = torch.split(tensor, tensor.size(-1) // split_amount, dim=-1)  # List of Tensors

    # Apply log operation if log_albedo is True
    if log_albedo:
        tensors_list[-1] = log_tensor(tensors_list[-1])  # Apply logTensor to the last element
        tensors_list[1] = log_tensor(tensors_list[1])  # Apply logTensor to the second element

    # Stack the tensors along a new dimension (axis=1)
    tensors = torch.stack(tensors_list, dim=1)  # Shape: [batch, split_amount, 256, 256, channels/split_amount]

    # Reshape the tensor
    shape = list(tensors.size())  # Current shape: [batch, split_amount, 256, 256, channels/split_amount]
    new_shape = [shape[0] * shape[1]] + shape[2:]  # Flatten batch and split_amount dimensions
    tensors_reshaped = tensors.view(new_shape)  # Reshape tensor to [batch * split_amount, 256, 256, channels/split_amount]

    return tensors_reshaped

# Usage example:
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seed for reproducibility
    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    # Create output directory if it doesn't exist
    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    # Load options from the checkpoint if in test/export/eval mode
    if a.mode in ["test", "export", "eval"]:
        if a.checkpoint is None:
            raise Exception("Checkpoint required for test, export, or eval mode")
        
        # Load options from JSON file
        options = {"which_direction", "ngf", "ndf", "nbTargets", "depthFactor", "loss", "useLog"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("loaded", key, "=", val)
                    setattr(a, key, val)

        # Disable certain features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    # Print all arguments
    for k, v in vars(a).items():
        print(k, "=", v)

    # Save options to output directory
    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    # Load datasets
    examples = load_examples(a.input_dir, a.mode == "train")
    print(f"{a.mode} set count = {len(examples)}")
    eval_examples = None
    if a.mode == "train":
        eval_dir = os.path.join(os.path.dirname(a.input_dir), "testBlended")
        eval_examples = load_examples(eval_dir, False)
        print(f"Evaluation set count = {len(eval_examples)}") 
    
    # Create model
    model = create_model(device)
    
    if a.mode == "train":
        model_test = create_model()

    # Prepare optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr, betas=(0.5, 0.999))
    criterion = torch.nn.L1Loss()

    # Use DataLoader for batching
    train_loader = DataLoader(examples, batch_size=a.batch_size, shuffle=a.mode == "train")
    eval_loader = DataLoader(eval_examples, batch_size=a.batch_size, shuffle=False) if eval_examples else None

    # Prepare tensorboard writer
    writer = SummaryWriter(log_dir=a.output_dir) if a.summary_freq > 0 else None

    # Load checkpoint if available
    if a.checkpoint is not None:
        print(f"Loading model from checkpoint: {a.checkpoint}")
        checkpoint = torch.load(os.path.join(a.checkpoint, "model.pth"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Training or testing
    if a.mode in ["test", "eval"]:
        print("Running in test/eval mode")
        with torch.no_grad():
            run_test(model, eval_loader, writer)
    else:
        print("Running in training mode")
        train(model, model_test, train_loader, eval_loader, optimizer, criterion, writer)

def train(model, model_test, train_loader, eval_loader, optimizer, criterion, writer):
    model.train()
    start_time = time.time()

    for epoch in range(a.max_epochs):
        for step, batch in enumerate(train_loader):
            inputs, targets = batch["inputs"], batch["targets"]

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step = epoch * len(train_loader) + step

            # Log progress
            if global_step % a.progress_freq == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

            # Write to tensorboard
            if writer and global_step % a.summary_freq == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)

            # Save checkpoint
            if global_step % a.save_freq == 0:
                save_checkpoint(model, optimizer, global_step)

            # Evaluate on test set
            if global_step % a.test_freq == 0 and eval_loader is not None:
                run_test(model_test, eval_loader, writer)

def run_test(model, data_loader, writer):
    model.eval()
    for step, batch in enumerate(data_loader):
        inputs, targets = batch["inputs"], batch["targets"]
        outputs = model(inputs)

        # Save images or log them
        save_images(outputs, step)
        if writer:
            writer.add_images("Outputs/test", outputs, step)

def save_checkpoint(model, optimizer, step):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    torch.save(checkpoint, os.path.join(a.output_dir, f"model_{step}.pth"))

# Normalizes a tensor throughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps the 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def torch_ormalize(tensor):
    # Compute the length along the channels dimension
    length = torch.sqrt(torch.sum(torch.square(tensor), dim=-1, keepdim=True))
    
    # Normalize the tensor by dividing each element by its corresponding length
    normalized_tensor = tensor / length
    return normalized_tensor

# Computes the dot product between 2 tensors (BatchSize, Width, Height, Channels)
# Keeps the 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def torch_DotProduct(tensorA, tensorB):
    return torch.sum(tensorA * tensorB, dim=-1, keepdim=True)

##########Rendering loss

def torch_render_diffuse_Substance(diffuse, specular):
    return diffuse * (1.0 - specular) / math.pi

def torch_render_D_GGX_Substance(roughness, NdotH):
    alpha = torch.square(roughness)
    underD = 1 / torch.maximum(torch.tensor(0.001), (torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0))
    return (torch.square(alpha * underD) / math.pi)

def torch_lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT * torch.square(distance))

def torch_render_F_GGX_Substance(specular, VdotH):
    sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
    return specular + (1.0 - specular) * sphg

def torch_render_G_GGX_Substance(roughness, NdotL, NdotV):
    return G1_Substance(NdotL, torch.square(roughness) / 2) * G1_Substance(NdotV, torch.square(roughness) / 2)

def G1_Substance(NdotW, k):
    return 1.0 / torch.maximum((NdotW * (1.0 - k) + k), torch.tensor(0.001))

def squeezeValues(tensor, min, max):
    return torch.clamp(tensor, min, max)

# svbrdf : (BatchSize, Width, Height, 4 * 3)
# wo : (BatchSize,1,1,3)
# wi : (BatchSize,1,1,3)
def torch_Render(svbrdf, wi, wo, includeDiffuse=True):
    wiNorm = torch_Normalize(wi)
    woNorm = torch_Normalize(wo)
    h = torch_Normalize((wiNorm + woNorm) / 2.0)
    
    diffuse = squeezeValues(deprocess(svbrdf[:, :, :, 3:6]), 0.0, 1.0)
    normals = svbrdf[:, :, :, 0:3]
    specular = squeezeValues(deprocess(svbrdf[:, :, :, 9:12]), 0.0, 1.0)
    roughness = squeezeValues(deprocess(svbrdf[:, :, :, 6:9]), 0.0, 1.0)
    roughness = torch.maximum(roughness, torch.tensor(0.001))
    
    NdotH = torch_DotProduct(normals, h)
    NdotL = torch_DotProduct(normals, wiNorm)
    NdotV = torch_DotProduct(normals, woNorm)
    VdotH = torch_DotProduct(woNorm, h)

    diffuse_rendered = torch_render_diffuse_Substance(diffuse, specular)
    D_rendered = torch_render_D_GGX_Substance(roughness, torch.maximum(torch.tensor(0.0), NdotH))
    G_rendered = torch_render_G_GGX_Substance(roughness, torch.maximum(torch.tensor(0.0), NdotL), torch.maximum(torch.tensor(0.0), NdotV))
    F_rendered = torch_render_F_GGX_Substance(specular, torch.maximum(torch.tensor(0.0), VdotH))

    specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
    result = specular_rendered

    if includeDiffuse:
        result = result + diffuse_rendered

    lampIntensity = 1.0
    # lampDistance = torch.sqrt(torch.sum(torch.square(wi), dim=3, keepdim=True))

    lampFactor = lampIntensity * math.pi  # torch_lampAttenuation(lampDistance) * lampIntensity * math.pi

    result = result * lampFactor

    result = result * torch.maximum(torch.tensor(0.0), NdotL) / torch.unsqueeze(
        torch.maximum(wiNorm[:, :, :, 2], torch.tensor(0.001)), dim=-1
    )  # This division compensates for the cosine distribution of intensity in rendering

    return [result, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]


main()