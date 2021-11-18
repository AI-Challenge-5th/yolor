from models.models import *
from utils.torch_utils import select_device, time_synchronized, prune, sparsity
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prune weight")
    parser.add_argument('--prune', type=float, default=0.20)
    args = parser.parse_args()
    
    model = Darknet('cfg/yolor_p6.cfg')
    ckpt = torch.load("yolor_p6.pt")  # load checkpoint
    ckpt['model'] = {k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
    model.load_state_dict(ckpt['model'], strict=False)
    prune(model, args.prune)
    torch.save({'model':model.state_dict()}, "yolor_p6_pr{}.pt".format(int(args.prune*100)))