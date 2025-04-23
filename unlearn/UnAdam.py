from trainer import untrain

from .impl import iterative_unlearn


@iterative_unlearn
def UnAdam(data_loaders, sub_datasets, model, criterion, optimizer, epoch, args, mask):
    #retain_loader = data_loaders["retain"]
    return untrain.untrain(data_loaders, sub_datasets, model, criterion, optimizer, epoch, args, mask)
