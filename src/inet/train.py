import torch
import torch.nn as nn

class TrainLoop:
    """
    Traing function
    :param eq: the equation of the model
    :param unet: network for the trial solution of PDE $u$
    :param vnet: network for the Lagrangian multiplier $v$
    :param optim_u: the optimizer
    :param optim_v: the optimizer
    :dataloader: two dataloaders for interior and boundary data
    :num_epoches: the number of epoches for training
    :device: 'cuda' is use gpu
    :compute_err: default None, pass the error computation funtion
    :x_test: test data for computing the error
    :Area: boundary surface area, default 1
    :Vol: volume of the domain, default 1
    :use_vop: whether network v is defined with Laplacian or not, default True
    :use_scheduler: whether scheduler is used in the optimizer, default False
    :use_fullbatch: whether use full batch, default True
    :use_pinn: whether use PINN or not, default False
    """

    def __init__(
        self,
        eq,
        unet,
        vnet,
        optim_u, 
        optim_v,
        dataloader,
        num_epochs,
        device,
        compute_err = None,
        x_test = None,
        Area = 1,
        Vol = 1,
        u_scheduler = None,
        v_scheduler = None,
        use_vop = True,
        use_fullbatch = True,
        use_pinn = False
   ):
        self.eq = eq
        self.unet = unet
        self.vnet = vnet
        self.optim_u = optim_u 
        self.optim_v = optim_v
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.compute_err = compute_err
        self.x_test = x_test
        self.Area = Area
        self.Vol = Vol
        self.device = device
        self.scheduler_u = u_scheduler
        self.scheduler_v = v_scheduler
        self.use_vop = use_vop 
        self.use_fullbatch = use_fullbatch
        self.use_pinn = use_pinn
        self.losses = [] 
        self.err = []
        

    def run_loop(self):
        losses = []
        errs = []
        dataloader_x,dataloader_xb = self.dataloader
        if self.use_fullbatch:
            for ii, (x, xb) in enumerate(zip(dataloader_x,dataloader_xb)):
                break
            assert ii == 0
            x = x.to(self.device)
            xb = xb.to(self.device)
            for epoch in range(self.num_epochs):
                if self.use_pinn == False:
                    loss_u, loss_v, err = self.run_one(x,xb)
                else:
                    loss_u, loss_v, err = self.run_one_pinn(x,xb)
                if epoch%100==0 and self.compute_err is not None:
                    print('epoch: %d, loss: %f, err: %f'%(epoch,loss_u,err))
                elif epoch%100==0:
                    print('epoch: %d, loss:%f'%(epoch,loss_u))
                if self.scheduler_u:
                    self.scheduler_u.step()
                    self.scheduler_v.step()
                losses.append(loss_u)
                errs.append(err)
        else:
            for epoch in range(self.num_epochs):
                dataloader_x,dataloader_xb = self.data
                loss_epoch = 0 
                err_epoch = 0
                for ii, (x, xb) in enumerate(zip(dataloader_x,dataloader_xb)):
                    x = x.to(self.device)
                    xb = xb.to(self.device)
                    if self.use_pinn == False:
                        loss_u, loss_v, err = self.run_one(x,xb)
                    else:
                        loss_u, loss_v, err = self.run_one_pinn(x,xb)
                    loss_epoch += loss_u
                    if self.compute_err is not None:
                        err_epoch += err
                if self.scheduler_u:
                    self.scheduler_u.step()
                    self.scheduler_v.step()

                losses.append(loss_epoch/(ii+1))
                errs.append(err_epoch/(ii+1))
                if epoch%100==0 and self.compute_err is not None:
                    print('epoch: %d, loss: %f, err: %f'%(epoch,loss_epoch/(ii+1),err_epoch/(ii+1)))
                elif epoch%100==0:
                    print('epoch: %d, loss:%f'%(epoch,loss_epoch/(ii+1)))
        self.losses = losses
        self.err = errs
    

    def run_one_pinn(self,x,xb):
        # Train u, one step
        x.requires_grad = True
        self.unet.zero_grad()
        u = self.unet(x)
        u_xx = self.eq.opA(u,x)
        xb.requires_grad = True
        ub = self.unet(xb)
        ubB = self.eq.opB(ub,xb)
        loss_u = (0.5*torch.mean((ubB-self.eq.g(xb))**2)*self.Area + torch.mean((- u_xx - self.eq.f(x))**2)*self.Vol)
        lossu = loss_u.item()
        loss_u.backward()
        self.optim_u.step()
        err = None 
        if self.compute_err is not None:
            err = self.compute_err(self.unet, self.eq.ur, self.x_test.to(self.device)).item()
        return lossu, None, err

    def run_one(self,x,xb):
        # Train u, one step
        x.requires_grad = True
        self.unet.zero_grad()
        u = self.unet(x)
        u_xx = self.eq.opA(u,x)
        v = self.vnet(x)
        v = self.eq.opA(v,x)
        xb.requires_grad = True
        ub = self.unet(xb)
        ubB = self.eq.opB(ub,xb)
        loss_u = (0.5*torch.mean((ubB-self.eq.g(xb))**2)*self.Area + torch.mean((- u_xx - self.eq.f(x))*v.detach())*self.Vol)
        lossu = loss_u.item()
        loss_u.backward()
        self.optim_u.step()
        #Train v, one step
        self.vnet.zero_grad()
        u = self.unet(x)
        v = self.vnet(x)
        v = self.eq.opA(v,x)
        u_xx = self.eq.opA(u,x)
        loss_v = -torch.mean((-u_xx.detach() - self.eq.f(x))*v)*self.Vol
        lossv = loss_v.item()
        loss_v.backward()
        self.optim_v.step()
        with torch.no_grad():
            for param in self.vnet.parameters():
                param.clamp_(-0.1, 0.1)

        #compute err
        err = None 
        if self.compute_err is not None:
            err = self.compute_err(self.unet, self.eq.ur, self.x_test.to(self.device)).item()
        return lossu, lossv, err

    
    def save_model(self, filename):
        torch.save(self.unet.state_dict(), filename)

    def load_model(self,filename):
        checkpoint = torch.load(filename)
        self.unet.load_state_dict(checkpoint)
