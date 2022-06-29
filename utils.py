import shutil
import torch
import cv2
import numpy as np


def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def hw2heatmap(x: torch.Tensor, size = 256):
    device = x.device
    assert len(x.shape) == 2
    map: np.ndarray = cam(x.detach().cpu().numpy(), size)
    map = map[:,:,::-1].copy() * 2 - 1
    return torch.from_numpy(map).permute(2, 0, 1).contiguous().to(device)

def bhw2heatmap(bx: torch.Tensor, size = 256):
    ans = []
    for i in range(bx.size(0)):
        ans.append(hw2heatmap(bx[i][0]))
    return torch.stack(ans, dim=0)

def image_blend_normal(img1: torch.Tensor, img2: torch.Tensor, alpha_a: float = 0.5):
    return img1 * alpha_a + img2 * (1 - alpha_a)

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
