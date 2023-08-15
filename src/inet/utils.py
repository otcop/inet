import torch
def compute_err(unet,ur,x):
    u_pred = unet(x)[:,0]
    u_true = ur(x)
    return torch.linalg.norm(u_true - u_pred) / torch.linalg.norm(u_true)
