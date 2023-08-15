"""Define the model"""
import torch
import torch.nn as nn

class MODEL(nn.Module):
  """
  The MLP for representing solutions of PDEs and the Lagrangian multiplier
  :param d: dimensions of the domain
  :param hdim: number of neurons in hidden layers
  :param depth: number of hidden layers
  :param act: activation function, default tanh
  """
  def __init__(self, d=2, hdim=20, depth=1, act=nn.Tanh()):
    super().__init__()
    self.d = d
    self.hdim = hdim
    self.depth = depth
    self.act = act
    layers = nn.ModuleList()
    layers.append(nn.Linear(d,hdim))
    layers.append(act)
    for i in range(depth):
      layers.append(nn.Linear(hdim,hdim))
      layers.append(act)
    layers.append(nn.Linear(hdim,1))


    self.l = nn.Sequential(*layers)
    # self.lp = nn.Linear(d,hdim)
    # self.out = nn.Linear(hdim,1)
    self.out = nn.Linear(d,1,bias=False)


  def forward(self, x):
    # return self.out(self.l(x)+self.lp(x))
    return self.l(x) #+ self.out(x)
class Unet(nn.Module):
    pass
if __name__ == "__main__":
  model = MODEL()
  x = torch.randn((100,2))
  print(model(x).shape)
  print(model)
