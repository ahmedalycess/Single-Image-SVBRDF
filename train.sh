#!/bin/bash

input_dir="./DeepMaterialsData/Data_Deschaintre18/trainBlended"
model_dir="./model"
epochs=500
 
python main.py --mode train --input-dir $input_dir --model-dir $model_dir --epochs $epochs --save-frequency 50 --validation-frequency 1 --retrain --batch-size 10