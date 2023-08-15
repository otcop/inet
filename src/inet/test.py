import torch
def load_data(d, num_int, num_ext):
  """
  Mento carlo sampling on -1--1
  :param d: dimension
  :param num_int: number of interior sampling points
  :param num_ext: number of exterior sampling points
  :param batch_size: batch size
  Return:
    two data loader, first for the interior points, second for the exterior points
  """

  x_dist = torch.distributions.Uniform(-1., 1.)
  xs = x_dist.sample((num_int,d))
  xb = x_dist.sample((2*d,num_ext,d))
  for dd in range(d):
    xb[dd,:,dd] = torch.ones(num_ext)
    xb[dd + d,:,dd] = -torch.ones(num_ext)
  xb = xb.reshape(2*d*num_ext,d)
  return xs,xb
