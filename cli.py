import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Single-Image-SVBRDF-Capture-rendering-loss')

    parser.add_argument('--mode', '-M', dest='mode', action='store', required=True, choices=['train', 'test'], default='train')

    parser.add_argument('--input-dir', '-i', dest='input_dir', action='store', required=True)

    parser.add_argument('--model-dir', '-m', dest='model_dir', action='store', required=True, help='Directory for the model and training metadata.')

    parser.add_argument('--save-frequency', dest='save_frequency', action='store', required=False, type=int, choices=range(1, 1000), default=50, metavar="[0-1000]",
                        help='Number of consecutive training epochs after which a checkpoint of the model is saved. Default is %(default)s.')
    
    parser.add_argument('--validation-split', dest='validation_split', action='store', required=False, type=float, default=0.1, help='Fraction of the training data to use for validation. Default is %(default)s.')
    
    parser.add_argument('--batch-size', '-b', dest='batch_size', action='store', required=False, type=int, default=8, help='Batch size for training and validation. Default is %(default)s.')

    parser.add_argument('--learning-rate', '-lr', dest='learning_rate', action='store', required=False, type=float, default=1e-5, help='Learning rate for the optimizer. Default is %(default)s.')

    parser.add_argument('--l1-weight', dest='l1_weight', action='store', required=False, type=float, default=0.1, help='Weight for the L1 loss. Default is %(default)s.')

    parser.add_argument('--l2-weight', dest='l2_weight', action='store', required=False, type=float, default=0.1, help='Weight for the L2 loss. Default is %(default)s.')

    parser.add_argument('--validation-frequency', dest='validation_frequency', action='store', required=False, type=int, choices=range(1, 1000), default=25, metavar="[0-1000]",
                        help='Number of consecutive training epochs after which validation is performed. Default is %(default)s.')

    parser.add_argument('--epochs', '-e', dest='epochs', action='store', type=int, default=100, help='Maximum number of epochs to run the training for.')

    parser.add_argument('--retrain', dest='retrain', action='store_true', default=False, help='When training, ignore any data in the model directory.')

    args = parser.parse_args()

    return args