#!/bin/bash

python3 material_net_pytorch.py --mode train --output_dir pytorch_output_dir --input_dir DeepMaterialsData/Data_Deschaintre18/trainBlended --batch_size 8 --loss l1 --max_epochs 1 --useLog
