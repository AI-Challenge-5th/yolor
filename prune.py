from models.models import *
from utils.torch_utils import select_device, time_synchronized, prune, sparsity


if __name__ == "__main__":
    model = Darknet('cfg/yolor_p6.cfg')
    ckpt = torch.load("yolor_p6.pt")  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    prune(model, 0.20)
    torch.save({'model':model.state_dict()}, "yolor_p6_pr20.pt")