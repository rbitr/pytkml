import torch
from torch.utils.data import DataLoader, random_split


def RandomSubsetLoader(dataset, num, batch_size=16,seed=0):
    """Returns a Dataloader with num random samples from dataset
    """

    def get_loader():
        subset, _ = random_split(dataset,[num, len(dataset)-num],generator=torch.Generator().manual_seed(seed))
        return DataLoader(subset,batch_size=batch_size,shuffle=False)

    return get_loader
