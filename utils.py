import shutil

import torch


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth')

def computeModelParametersNorm1(model: torch.nn.Module):
    parameters_norm = 0
    nparameters = 0
    for param in model.parameters():
        paramd = param.detach()
        parameters_norm += torch.norm(paramd, 1)
        nparameters += paramd.numel()
    return parameters_norm, nparameters

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
