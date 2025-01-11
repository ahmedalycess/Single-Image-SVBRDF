import math
from torch.utils.tensorboard import SummaryWriter
import shutil
import torch

from cli import parse_args
from dataset import MaterialDataset
from losses import MixedLoss
from model import SVBRDFNetwork
from pathlib import Path
from persistence import Checkpoint
from renderer import LocalRenderer
import utils
import scene

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "expandable_segments:True"

args = parse_args()

clean_training = args.mode == 'train' and args.retrain

# Load the checkpoint 
checkpoint_dir = Path(args.model_dir)
checkpoint = Checkpoint()
if not clean_training:
    checkpoint = Checkpoint.load(checkpoint_dir)

# Make the result reproducible
utils.enable_deterministic_random_engine()

# Determine the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Create the model
model = SVBRDFNetwork().to(device)
if checkpoint.is_valid():
    model = checkpoint.restore_model_state(model)
elif args.mode == 'test':
    print("No model found in the model directory but it is required for testing.")
    exit(1)


data = MaterialDataset(data_directory=args.input_dir, image_size=256)

epoch_start = 0
if checkpoint.is_valid():
    epoch_start = checkpoint.restore_epoch(epoch_start)

if args.mode == 'train':
    validation_split = args.validation_split
    print("Using {:.2f} % of the data for validation".format(round(validation_split * 100.0, 2)))
    training_data, validation_data = torch.utils.data.random_split(data, [int(math.ceil(len(data) * (1.0 - validation_split))), int(math.floor(len(data) * validation_split))])
    print("Training samples: {:d}.".format(len(training_data)))
    print("Validation samples: {:d}.".format(len(validation_data)))

    training_dataloader   = torch.utils.data.DataLoader(training_data,   batch_size=args.batch_size, shuffle=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    batch_count = int(math.ceil(len(training_data) / training_dataloader.batch_size))

    epoch_end = args.epochs

    print("Training from epoch {:d} to {:d}".format(epoch_start, epoch_end))

    # Set up the optimizer
    optimizer     = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    if checkpoint.is_valid():
        optimizer = checkpoint.restore_optimizer_state(optimizer)

    loss_renderer = None
    loss_renderer = LocalRenderer()

    loss_function = MixedLoss(loss_renderer, l1_weight=args.l1_weight, l2_weight=args.l2_weight)

    # Setup statistics stuff
    statistics_dir = checkpoint_dir / "logs"
    if clean_training and statistics_dir.exists():
        # Nuke the stats dir
        shutil.rmtree(statistics_dir)
    statistics_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(statistics_dir.absolute()))
    last_batch_input = None

    # Clear checkpoint in order to free up some memory
    checkpoint.purge()

    model.train()
    for epoch in range(epoch_start, epoch_end):
        for i, batch in enumerate(training_dataloader):
            # Unique index of this batch
            batch_index = epoch * batch_count + i

            # Construct inputs
            batch_input = batch["input"].to(device)
            batch_svbrdf = batch["svbrdf"].to(device)

            # Perform a step
            optimizer.zero_grad() 
            outputs = model(batch_input)
            loss    = loss_function(outputs, batch_svbrdf)
            loss.backward()
            optimizer.step()

            print("Epoch {:d}, Batch {:d}, loss: {:f}".format(epoch, i + 1, loss.item()))
            # Statistics
            writer.add_scalar("loss/train", loss.item(), batch_index)
            last_batch_input = batch_input

        if epoch % args.save_frequency == 0:
            Checkpoint.save(checkpoint_dir, args, model, optimizer, epoch)

        if epoch % args.validation_frequency == 0 and len(validation_data) > 0:
            model.eval()
            
            val_loss = 0.0
            batch_count_val = 0
            for batch in validation_dataloader:
                # Construct inputs
                batch_input = batch["input"].to(device)
                batch_svbrdf = batch["svbrdf"].to(device)

                outputs  = model(batch_input)
                val_loss += loss_function(outputs, batch_svbrdf).item()
                batch_count_val += 1
            val_loss /= batch_count_val

            print("Epoch {:d}, validation loss: {:f}".format(epoch, val_loss))
            writer.add_scalar("loss/validation", val_loss, (epoch + 1) * batch_count)
        
            model.train()

    # Save a final snapshot of the model
    Checkpoint.save(checkpoint_dir, args, model, optimizer, epoch)

    writer.add_graph(model, last_batch_input) 
    writer.close()

    # Use the validation dataset as test data
    if len(validation_data) == 0:
        # Fixed fallback if the training set is too small
        print("Training dataset too small for validation split. Using training data for validation.")
        validation_data = training_data

    # Use the validation dataset as test data
    test_data = validation_data 
else:
    test_data = data

model.eval()

# pick a random sample from the test data
import random
random_index = random.randint(0, len(test_data) - 1)
print("Rendering sample {:d} from the test data.".format(random_index))
test_data = torch.utils.data.Subset(test_data, [random_index]) 

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, pin_memory=True)
renderer = LocalRenderer()
scene    = scene.Scene(scene.View([0.0, -1.0, 2.0]), scene.Light([0.0, 0.0, 2.0], [50.0, 50.0, 50.0]))
# Plotting
import matplotlib.pyplot as plt

fig=plt.figure()
row_count = 2
col_count = 6
for i_row, batch in enumerate(test_dataloader):
    # Construct inputs
    batch_input = batch["input"].to(device)
    batch_svbrdf = batch["svbrdf"].to(device)

    outputs = model(batch_input)

    input       = utils.gamma_encode(batch_input.squeeze(0)).cpu().permute(1, 2, 0)
    target_maps = torch.cat(batch_svbrdf.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)
    output_maps = torch.cat(outputs.split(3, dim=1), dim=0).clone().cpu().detach().permute(0, 2, 3, 1)

    fig.add_subplot(row_count, col_count, 1)
    plt.imshow(input)
    plt.axis('off')


    rendering_target = renderer.render(scene, batch_svbrdf)    
    rendering_target = utils.gamma_encode(rendering_target).clone().cpu().detach().squeeze(0).permute(1, 2, 0)
    fig.add_subplot(row_count, col_count, 2)
    plt.imshow(rendering_target)
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 3)
    plt.imshow(utils.deprocess(target_maps[0]))
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 4)
    plt.imshow(target_maps[1])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 5)
    plt.imshow(target_maps[2])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 6)
    plt.imshow(target_maps[3])
    plt.axis('off')


    rendering_output = renderer.render(scene, outputs)
    rendering_output = utils.gamma_encode(rendering_output).clone().cpu().detach().squeeze(0).permute(1, 2, 0)
    fig.add_subplot(row_count, col_count, 8)
    plt.imshow(rendering_output)
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 9)
    plt.imshow(utils.deprocess(output_maps[0]))
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 10)
    plt.imshow(output_maps[1])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 11)
    plt.imshow(output_maps[2])
    plt.axis('off')

    fig.add_subplot(row_count, col_count, 12)
    plt.imshow(output_maps[3])
    plt.axis('off')
plt.show()
