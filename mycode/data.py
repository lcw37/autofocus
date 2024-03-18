import torchvision.transforms
from train import Noise, DatasetFromHdf5, grayloader
from torch.utils.data import DataLoader


def get_transform(noise_prob=0.0, noise_type=None, rescale=False):
    return torchvision.transforms.Compose(
        [Noise(probability=noise_prob, noise_type=noise_type)]
    )


def get_dataset_loader(h5_filepath, loader, transform, batch_size):
    dataset = DatasetFromHdf5(h5_filepath,
                              loader=grayloader,
                              transform=transform)
    loader = DataLoader(dataset=dataset, 
                        batch_size=batch_size, 
                        shuffle=True,
                        num_workers=0)
    return loader