import gc
import pathlib
import torch

class Checkpoint:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    @staticmethod
    def get_checkpoint_path(checkpoint_dir):
        return checkpoint_dir.joinpath("checkpoint.tar")

    @classmethod
    def load(cls, checkpoint_dir):
        if not isinstance(checkpoint_dir, pathlib.Path):
            checkpoint_dir = pathlib.Path(checkpoint_dir)
        
        checkpoint_path = Checkpoint.get_checkpoint_path(checkpoint_dir)

        if not checkpoint_path.exists():

            print("No checkpoint found in directory '{}'".format(checkpoint_dir))
            return cls(None)

        return cls(torch.load(checkpoint_path))

    @staticmethod
    def save(checkpoint_dir, args, model, optimizer, epoch):
        if not isinstance(checkpoint_dir, pathlib.Path):
            checkpoint_dir = pathlib.Path(checkpoint_dir)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
        }

        checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        torch.save(checkpoint, Checkpoint.get_checkpoint_path(checkpoint_dir))

    def purge(self):
        self.checkpoint = None
        gc.collect()

    def is_valid(self):
        return self.checkpoint is not None

    def restore_model_state(self, model):
        if 'model_state_dict' in self.checkpoint:
            model.load_state_dict(self.checkpoint['model_state_dict'])
            print("Restored model state")
        else:
            print("Failed to restore model state")

        return model

    def restore_epoch(self, epoch):
        if 'epoch' in self.checkpoint:
            epoch = self.checkpoint['epoch']
            print("Restored epoch {}".format(epoch))
        else:
            print("Failed to restore epoch")
        
        return epoch

    def restore_optimizer_state(self, optimizer):
        if 'optimizer_state_dict' in self.checkpoint:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            print("Restored optimizer state")
        else:
            print("Failed to restore optimizer state")

        return optimizer
