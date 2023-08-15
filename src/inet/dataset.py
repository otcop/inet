import torch

def load_data(d, num_int, num_ext, box= [-1,1]):
  """
  Mento carlo sampling on a box
  :param d: dimension
  :param num_int: number of interior sampling points
  :param num_ext: number of exterior sampling points
  :param box: [-L, L] 
  Return:
    two dataset, first for the interior points, second for the exterior points
  """

  x_dist = torch.distributions.Uniform(box[0], box[1])
  xs = x_dist.sample((num_int,d))
  num_ext = num_ext // (2*d)
  xb = x_dist.sample((2*d,num_ext,d))
  for dd in range(d):
    xb[dd,:,dd] = torch.ones(num_ext)*box[1]
    xb[dd + d,:,dd] =  torch.ones(num_ext)*box[0]
  xb = xb.reshape(2*d*num_ext,d)
  # dataloader_x = torch.utils.data.DataLoader(xs, batch_size=batch_size, shuffle=True,num_workers=num_workers)
  # dataloader_xb = torch.utils.data.DataLoader(xb, batch_size=int(2*d*num_ext/(num_int/batch_size)), shuffle=True,num_workers=num_workers)
  # print(xs.shape, xb.shape)
  # assert len(xs)//batch_size == len(xb) // int(2*d*num_ext/(num_int/batch_size))
  # return dataloader_x, dataloader_xb
  return xs, xb

def load_data_Lshape(d, num_int, num_ext, box = [-1,1], inner_box=[0,1]):
  """
  Mento Carlo sampling on [A,B]^d \ [C,B]^d
  :param d: dimension
  :num_int: number of interior
  :param num_int: number of interior sampling points
  :param num_ext: number of exterior sampling points
  :param box: [A,B]
  :param inner_box: [C,B]
  Return:
    two dataset, first for the interior points, second for the exterior points
  """

  assert box[1] == inner_box[1]
  base_dist0 = torch.distributions.uniform.Uniform(box[0],box[1])
  base_dist1 = torch.distributions.uniform.Uniform(inner_box[0],inner_box[1])
  # Sampling interior points
  x = base_dist0.sample((2*num_int,d))
  ind = torch.prod(x > inner_box[0], -1)==0
  x = x[ind,:]
  if len(ind) < num_int:
      x = x
  else:
      x = x[:num_int,:]
  # Sampling points on the boundary
  num_ext = num_ext // (2*d)
  xb = base_dist0.sample((2*d,2*num_ext,d))
  for dd in range(d):
      xb[dd,:,dd] = torch.ones(2*num_ext) * box[1]
      xb[dd + d,:,dd] =  torch.ones(2*num_ext) * box[0]
  xb = xb.reshape(2*d*2*num_ext,d)
  ind = torch.prod(xb > inner_box[0], -1)==0
  xb = xb[ind,:]
  xb1 = base_dist1.sample((2*d,num_ext,d))
  for dd in range(d):
      xb1[dd,:,dd] = torch.ones(num_ext) * box[1]
      xb1[dd + d,:,dd] =  torch.zeros(num_ext) * box[0]
  xb1 = xb1.reshape(2*d*num_ext,d)
  ind = torch.prod(xb1 > inner_box[0], -1)==0
  xb1 = xb1[ind,:]
  xb = torch.concat([xb,xb1])
  idx = torch.randperm(xb.shape[0])
  xb = xb[idx].view(xb.size())
  if len(xb) < num_ext*2*d:
    xb = xb 
  else:
    xb = xb[:num_ext*2*d,:]
  return x, xb




# def load_L_data2d(num_int, num_ext, batch_size):
#     """
#     Mento Carlo sampling on the L-shape data 
#     (-1,-1)->(1,-1)->(1,0)->(0,0)->(0,1)->(-1,1)->(-1,-1)
#     Return:
#       two dataset, first for interior points, second for boundary data points
#     """
#     d = 2
#     base_dist1 = torch.distributions.uniform.Uniform(0,1)
#     base_dist0 = torch.distributions.uniform.Uniform(-1,0)
#     x = base_dist0.sample((int(num_int/3),d))
#     xx1 = torch.cat([base_dist0.sample((int(num_int/3),1)),base_dist1.sample((int(num_int/3),1))],-1)
#     xx2 = torch.cat([base_dist1.sample((int(num_int/3),1)),base_dist0.sample((int(num_int/3),1))],-1)
#     x = torch.concat([x,xx1,xx2])
#     bsize=num_ext
#     x1 = torch.cat([base_dist0.sample((bsize,1)),torch.ones(bsize,1)],-1)
#     x2 = torch.cat([base_dist0.sample((bsize,1)),-torch.ones(bsize,1)],-1)
#     x3 = torch.cat([-torch.ones(bsize,1),base_dist0.sample((bsize,1))],-1)
#     x4 = torch.cat([-torch.ones(bsize,1),base_dist1.sample((bsize,1))],-1)
#     x5 = torch.cat([torch.ones(bsize,1),base_dist0.sample((bsize,1))],-1)
#     x6 = torch.cat([torch.zeros(bsize,1),base_dist1.sample((bsize,1))],-1)
#     x7 = torch.cat([base_dist1.sample((bsize,1)),-torch.ones(bsize,1)],-1)
#     x8 = torch.cat([base_dist1.sample((bsize,1)),torch.zeros(bsize,1)],-1)
#     xb = torch.concat([x1,x2,x3,x4,x5,x6,x7,x8])
#     dataloader_x = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)
#     dataloader_xb = torch.utils.data.DataLoader(xb, batch_size=len(xb)//(len(x)//batch_size), shuffle=True)
#     assert len(x)//batch_size == len(xb) // (len(xb)//(len(x)//batch_size))
#     return dataloader_x, dataloader_xb

if __name__ == "__main__":
  import matplotlib.pyplot as plt
  xs, xb = load_data(3, 1024, 32*6)
  print(xs.shape,xb.shape)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(xs[:,0],xs[:,1],xs[:,2])
  ax.scatter(xb[:,0],xb[:,1],xb[:,2])
  plt.show()
  xs, xb = load_data_Lshape(3,1024,32*6)
  print(xs.shape,xb.shape)
  fig = plt.figure()
  ax = fig.add_subplot(projection='3d')

  ax.scatter(xs[:,0],xs[:,1],xs[:,2])
  ax.scatter(xb[:,0],xb[:,1],xb[:,2])
  plt.show()
