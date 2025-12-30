import torch

def print_multiple(*args: torch.Tensor) :
    for arg in args :
        print(arg, arg.size())
    print()

# a tensor is a multidimentional matrix
# dim: 0, 1, ..., -2, -1

# boardcasting rules +, -, *, /
# 1. Right-alignment of shapes
#   The shapes of the two tensors are compared starting from the trailing (rightmost) dimension. 
#   Missing leading dimensions are implicitly treated as size 1.
# 2. Dimension-wise compatibility
#   For each aligned dimension pair, the sizes are compatible if they are equal or if one of them is 1.
# 3. Expansion of singleton dimensions
#   Any dimension with size 1 is conceptually expanded (without copying data) to match the 
#   other size in that dimension.
# 4. Resulting shape determination
#   The output tensor shape is the element-wise maximum along each aligned dimension.
# 5. Failure condition
#   If, in any aligned dimension, the sizes differ and neither is 1, broadcasting is undefined 
#   and a runtime error is raised.

# |s| = m, |t| = n
# s_i = t_i or s_i == 1 or t_i == 1
# resulting shape [max(s_1, t_1), ..., max(s_k, t_k)]   (k = max(m, n))

# it is hard for me to get my head into understanding what is actually going on
# therefore, most of the time i just look at what output dimention i need and 
# choose the dim based on that. it works almost always

# creating tensors
# inputs
a = torch.zeros(3)                      # [3]
b = torch.zeros(2, 3, 4)                # [2, 3, 4]

c = torch.ones(3)                       # [3]
d = torch.ones(2, 3, 4)                 # [2, 3, 4]

e = torch.tensor([1, 2, 3])             # [3]
f = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])                                      # [2, 2, 2]

g = torch.arange(6)                     # [6]

h = torch.rand(3)                       # [3]
i = torch.rand(2, 3, 4)                 # [2, 3, 4]
print_multiple(a, b, c, d, e, f, g, h, i)


# tensor.size
# inputs
a = torch.zeros(6)                      # [6]
b = torch.zeros(2, 3, 4)                # [2, 3, 4]
u = a.size()                            # [6]
v = b.size()                            # [2, 3, 4]


# tensor.reshape
a = torch.arange(12)                    # [12]
b = torch.zeros(2, 3, 4)                # [2, 3, 4]

u = a.reshape(2, 2, 3)                  # [2, 2, 3]
v = a.reshape(2, -1, 3)                 # [2, 2, 3]
w = b.reshape(-1)                       # [24]
print_multiple(a, b, u, v, w)


# tensor.unsqueeze
a = torch.zeros(2, 3)                   # [2, 3]

u = a.unsqueeze(0)                      # [1, 2, 3]
v = a[None]                             # [1, 2, 3]
w = a.unsqueeze(1)                      # [2, 1, 3]
x = a[:, None]                          # [2, 1, 3]
y = a.unsqueeze(2)                      # [2, 3, 1]
z = a[:, :, None]                       # [2, 3, 1]
print_multiple(a, u, v, w, x, y, z)


# tensor.repeat vs tensor.expand
a = torch.tensor([[1, 2, 3]])           # [1, 3]

# repeat (copies data)
u = a.unsqueeze(0).repeat(2, 1, 1)      # [2, 1, 3]
# expand (view, shared memory)
v = a.unsqueeze(0).expand(2, 1, 3)      # [2, 1, 3]
print_multiple(a, u, v)


# torch.cat
a = torch.ones(2, 3)
b = torch.zeros(2, 3)

u = torch.cat([a, b], dim=0)            # [4, 3]
v = torch.cat([a, b], dim=1)            # [2, 6]
print_multiple(a, b, u, v)


# torch.stack
a = torch.arange(3)                     # [3]
b = torch.arange(3)                     # [3]

u = torch.stack([a, b], dim=0)          # [2, 3]
v = torch.stack([a, b], dim=1)          # [3, 2]
print_multiple(a, b, u, v)

a = torch.ones(2, 3)                    # [2, 3]
b = torch.ones(2, 3)                    # [2, 3]

u = torch.stack([a, b], dim=0)          # [2, 2, 3]
v = torch.stack([a, b], dim=1)          # [2, 2, 3]
w = torch.stack([a, b], dim=2)          # [2, 3, 2]
print_multiple(a, b, u, v, w)


# torch.sum
a = torch.ones(2, 3, 4)
u = a.sum(dim=0)                        # [3, 4]
v = a.sum(dim=1)                        # [2, 4]
w = a.sum(dim=2)                        # [2, 3]
print_multiple(a, u, v, w)


# torch.einsum
a = torch.tensor([1., 2., 3.])          # shape: [3]
b = torch.tensor([4., 5., 6.])          # shape: [3]
u = torch.einsum("i,i->", a, b)         # scalar
print_multiple(a, b, u)

a = torch.ones(2, 3)                    # shape: [2, 3]
b = torch.tensor([1., 2., 3.])          # shape: [3]
u = torch.einsum("ij,j->i", a, b)       # shape: [2]
print_multiple(a, b, u)

a = torch.ones(5, 2, 3)                 # shape: [5, 2, 3]
b = torch.ones(5, 3, 4)                 # shape: [5, 3, 4]
u = torch.einsum("bij,bjk->bik", a, b)  # shape: [5, 2, 4]
print_multiple(a, b, u)

a = torch.ones(2, 3, 4)                 # shape: [2, 3, 4]
u = torch.einsum("ijk->ij", a)          # shape: [2, 3] (equivalent to sum over dim=-1)
v = torch.einsum("ijk->ik", a)          # shape: [2, 4] (equivalent to sum over dim=1)
print_multiple(a, u, v)


# (.{40}).*# -> $1#