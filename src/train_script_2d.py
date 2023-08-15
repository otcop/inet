# from inet.dataset import  load_data
# from inet.equation import EllipticEQ, opA, opB
# from inet.model import MODEL
# from inet.train import TrainLoop
from inet import load_data, EllipticEQ, opA, opB, MODEL, TrainLoop
import argparse
import torch 
torch.manual_seed(1234)
import torch.nn as nn
import numpy as np
from torch.autograd import grad 
from inet.utils import compute_err
import matplotlib.pyplot as plt 
import mlflow 
import pandas as pd
default_dict = dict(
    d=2,
    num_int=int(1000),
    num_ext=int(40),
    hdim=20,
    depth=1,
    num_epoches=1000,
    device='cpu',
    batch_size=1000,
    folder = './result',
    optim = 'rmsprop',
    lr=2e-4,
    run='',
    fullbatch=True,
    training_data = ''
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}



if __name__ == "__main__":
    mlflow.autolog()
    mlflow.log_params(default_dict)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser,default_dict)
    args = parser.parse_args()
    keys = default_dict.keys()
    args=args_to_dict(args,keys)
    df = pd.read_csv(args['training_data'], index_col=None)
    print(df.iloc[0].to_dict())
    args.update(df.iloc[0].to_dict())
    print(args)
    args['num_ext'] = int(args['num_ext'])
    args['d'] = int(args['d'])
    args['num_int'] = int(args['num_int'])
    args['batch_size'] = int(args['batch_size'])

    #Load the data and make dataloaders
    x, xb = load_data(d=args['d'], num_int=args['num_int'], num_ext=args['num_ext'])
    dataloader_x = torch.utils.data.DataLoader(x, batch_size = args['batch_size'],shuffle=True)
    batch_size_xb = len(xb)// (len(x) // args['batch_size'])
    dataloader_xb = torch.utils.data.DataLoader(xb, batch_size = batch_size_xb,shuffle=True)
    dataloader = dataloader_x, dataloader_xb
    
    # Define the network and optimizer
    unet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh())
    vnet = MODEL(args['d'], args['hdim'], args['depth'], act = nn.Tanh())
    unet = unet.to(args['device'])
    vnet = vnet.to(args['device'])
    if args['optim'] == 'adam':
        uop = torch.optim.Adam(unet.parameters(), lr=args['lr'], betas=(0.5, 0.999))
        vop = torch.optim.Adam(vnet.parameters(), lr=args['lr'], betas=(0.5, 0.999))
    elif args['optim'] == 'rmsprop':
        uop = torch.optim.RMSprop(unet.parameters(), lr=args['lr'])#,weight_decay=0.1)
        vop = torch.optim.RMSprop(vnet.parameters(), lr=args['lr'])#,weight_decay=0.1)
    
    u_scheduler = torch.optim.lr_scheduler.StepLR(uop, step_size=3500, gamma=0.1)
    v_scheduler = torch.optim.lr_scheduler.StepLR(vop, step_size=3500, gamma=0.1)

    # Define the equation
    def f(x):
        d = args['d']
        return d*torch.pi**2/4*torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range(d)]),0)
    g = lambda x: 0
    ur = lambda x: torch.prod(torch.stack([torch.cos(torch.pi*x[:,k]/2) for k in range( args['d'])]),0)
    eq = EllipticEQ(args['d'], f, g, opA, opB, ur)
    x_test = 2*torch.rand((100000,args['d']))-1

    # train 
    a = TrainLoop(eq,
        unet,
        vnet,
        uop,
        vop,
        dataloader = dataloader,
        num_epochs = args['num_epoches'],
        device=args['device'],
        compute_err=compute_err, 
        x_test=x_test,
        Area=2**(args['d']-1)*2*args['d'],
        Vol=2**args['d'],
        use_pinn = False
        # u_scheduler=u_scheduler,
        # v_scheduler=v_scheduler
        )
    a.run_loop()

    # np.save(args['folder'] + "/err_%d"%args['d'],a.err)
    # np.save(args['folder'] + "/loss_%d"%args['d'],a.losses)
    # a.save_model(args['folder']+ '/model_%d.pth'%args['d'])
    fig = plt.figure()
    plt.semilogy(a.err)
    mlflow.log_metric("error", a.err[-1])
    mlflow.log_metric("loss", a.losses[-1])
    mlflow.log_figure(fig, "err.png")
    # plt.plot(a.err)
    # plt.show()
