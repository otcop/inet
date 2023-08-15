import torch
import torch.nn as nn

class TrainLoop:
    """
    Traing function
    :param infsupnet: the class of the model
    :param unet: network for the PDE solution
    :param vnet: network for the Lagrangian multiplier
    :param optim_u: the optimizer
    :param optim_u: the optimizer
    :dataloader_x: dataloader of interior sampling points
    :dataloader_xb: dataloader of boundary sampling points
    :Area: boundary surface area
    :Vol: volume of the domain
    :num_epoches: the number of epoches for training
    :device: 'cuda' is use gpu
    """

    def __init__(
        self,
        eq,
        unet,
        vnet,
        data,
        batch_size,
        lr_u,
        lr_v,
        num_epochs,
        compute_err = None,
        x_test = None,
        Area = 1,
        Vol = 1
   ):
        self.eq = eq
        self.unet = unet
        self.vnet = vnet
        self.data = data
        self.batch_size = batch_size 
        self.num_epochs = num_epochs
        self.compute_err = compute_err
        self.x_test = x_test
        self.Area = Area
        self.Vol = Vol
        self.optim_u = torch.optim.RMSprop(unet.parameters(),lr = lr_u) 
        self.optim_v = torch.optim.RMSprop(vnet.parameters(),lr = lr_v) 
        self.losses = []
        self.err = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.cuda.is_available():
            self.unet = self.unet.to('cuda')
            self.vnet = self.vnet.to('cuda')

        else:
            self.unet = self.unet.to('cpu')
            self.vnet = self.vnet.to('cpu')
    def run_loop(self):
        losses = []
        err = []
        scheduler_u = torch.optim.lr_scheduler.StepLR(self.optim_u,3300,0.1)
        scheduler_v = torch.optim.lr_scheduler.StepLR(self.optim_v,3300,0.1)
        box = torch.tensor([-1.,1.]).cuda()
        x_dist = torch.distributions.Uniform(box[0], box[1])
        num_int = int(1024*8)
        num_ext = 128
        d = 2
        ii=1
        x = x_dist.sample((num_int,d))
        xb = x_dist.sample((2*d,num_ext,d))
        for dd in range(d):
            xb[dd,:,dd] =1# torch.ones(num_ext)*box[1]
            xb[dd + d,:,dd] =-1#  torch.ones(num_ext)*box[0]
        xb = xb.reshape(2*d*num_ext,d)

        for epoch in range(self.num_epochs):
            dataloader_x,dataloader_xb = self.data
            loss_epoch = 0
            for _ in range(1):
            #for ii, (x, xb) in enumerate(zip(dataloader_x,dataloader_xb)):
                x = x.to(self.device)
                xb = xb.to(self.device)
                x.requires_grad = True
                self.unet.zero_grad()
                u = self.unet(x)
                u_xx = self.eq.opA(u,x)
                v = self.vnet(x)
                v = self.eq.opA(v,x)
                xb.requires_grad = True
                ub = self.unet(xb)
                ubB = self.eq.opB(ub,xb)
                loss_u = (0.5*torch.mean((ubB-self.eq.g(x))**2)*self.Area + torch.mean((- u_xx - self.eq.f(x))*v.detach())*self.Vol)
                self.losses.append(loss_u.item())
                loss_u.backward()
                self.optim_u.step()

                self.vnet.zero_grad()
                u = self.unet(x)
                v = self.vnet(x)
                v = self.eq.opA(v,x)
                u_xx = self.eq.opA(u,x)
                loss_v = -torch.mean((-u_xx.detach() - self.eq.f(x))*v)*self.Vol
                loss_v.backward()
                self.optim_v.step()
                loss_epoch += loss_u.item()
            if self.compute_err is not None:
                err.append(self.compute_err(self.unet, self.eq.ur, self.x_test.to(self.device)).item())
            losses.append(loss_epoch/(ii+1))
            scheduler_u.step()
            scheduler_v.step()

            if epoch%100==0 and self.compute_err is not None:
                print('epoch: %d, loss: %f, err: %f'%(epoch,loss_epoch/(ii+1),err[-1]))
            elif epoch%100==0:
                print('epoch: %d, loss:%f'%(epoch,loss_epoch/(ii+1)))
        self.losses = losses
        self.err = err
    def save_model(self, filename):
        torch.save(self.unet.state_dict(), filename)
    def load_model(self,filename):
        checkpoint = torch.load(filename)
        self.unet.load_state_dict(checkpoint)
