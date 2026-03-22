import torch
from nis_mc.functions.breit_wigner import BreitWigner1D
bw = BreitWigner1D()
x = torch.tensor([[0.5]], requires_grad=True)
f = lambda x: bw(x[:, 0])
y = f(x)
print('y:', y)
y.backward()
print('grad:', x.grad)
