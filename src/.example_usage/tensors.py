import torch

def f(*args: torch.Tensor) :
    for arg in args :
        print(arg, arg.size())
    print()

# it is hard for me to get my head into understanding what is actually going on
# therefore, most of the time i just look at what output dimention i need and 
# choose the dim based on that. it works almost always

print("creating tensors:")
a = torch.zeros(3)                      # shape: [3]
b = torch.ones(2, 3)                    # shape: [2, 3]
c = torch.ones(2, 3, 4)                 # shape: [2, 3, 4]
d = torch.tensor([[1, 2], [3, 4]])      # shape: [2, 2]
e = torch.arange(3)                     # shape: [3]
g = torch.rand(2, 3)                    # shape: [2, 3]
f(a, b, c, d, e, g)

print("\nconcatenation along an existing dimension:")
a = torch.zeros(3)                      # shape: [3]
b = torch.ones(2)                       # shape: [2]
x = torch.cat([a, b], dim=0)            # shape: [5]
f(a, b, x)

print("\ntargeting different dimentions:")
# a tensor is a multidimentional matrix
# dim: 0, 1, ..., -2, -1
a = torch.tensor([[1, 2], [3, 4]])      # shape: [2, 2]
b = torch.tensor([[5, 6], [7, 8]])      # shape: [2, 2]
x = torch.cat([a, b], dim=0)            # shape: [4, 2]
y = torch.cat([a, b], dim=1)            # shape: [2, 4]
f(a, b, x, y)

print("\nconcatenation along a new dimension:")
print("the given dimention is the 'new' dimention")
a = torch.tensor([1, 2, 3])             # shape: [3]
b = torch.tensor([5, 6, 7])             # shape: [3]
x = torch.stack([a, b], dim=0)          # shape: [2, 3]
y = torch.stack([a, b], dim=1)          # shape: [3, 2]
f(a, b, x, y)

print("\nreduction along dimensions:")
print("the given dimention is 'deleted (summed over)'")
a = torch.tensor([[1, 2, 3], [4, 5, 6]])# shape: [2, 3]
x = torch.sum(a, dim=0)                 # shape: [3]
y = torch.sum(a, dim=1)                 # shape: [2]
f(a, x, y)

print("\nintroducing a singleton dimension:")
a = torch.zeros(2, 3)                   # shape: [2, 3]
x = a[:, :, None]                       # shape: [2, 3, 1]
y = a[:, None, :]                       # shape: [2, 1, 3]
u = a[:, None]                          # shape: [2, 1, 3]
z = a[None, :, :]                       # shape: [1, 2, 3]
v = a[None]                             # shape: [1, 2, 3]
f(a, x, y, u, z, v)

print("\nexplicit tensor contraction across dimensions:")
a = torch.tensor([1., 2., 3.])          # shape: [3]
b = torch.tensor([4., 5., 6.])          # shape: [3]
x = torch.einsum("i,i->", a, b)         # scalar
f(a, b, x)
M = torch.ones(2, 3)                    # shape: [2, 3]
v = torch.tensor([1., 2., 3.])          # shape: [3]
x = torch.einsum("ij,j->i", M, v)       # shape: [2]
f(M, v, x)
A = torch.ones(5, 2, 3)                 # shape: [5, 2, 3]
B = torch.ones(5, 3, 4)                 # shape: [5, 3, 4]
x = torch.einsum("bij,bjk->bik", A, B)  # shape: [5, 2, 4]
f(A, B, x)
x = torch.ones(2, 3, 4)                 # shape: [2, 3, 4]
y = torch.einsum("ijk->ij", x)          # shape: [2, 3] (equivalent to sum over dim=-1)
z = torch.einsum("ijk->ik", x)          # shape: [2, 4] (equivalent to sum over dim=1)
f(x, y, z)

print("\nintroducing a singleton dimension with unsqueeze:")
print("the given dimension specifies the index at which a size-1 axis is inserted")
a = torch.zeros(2, 3)                    # shape: [2, 3]
x = a.unsqueeze(0)                       # shape: [1, 2, 3]
y = a.unsqueeze(1)                       # shape: [2, 1, 3]
z = a.unsqueeze(2)                       # shape: [2, 3, 1]
f(a, x, y, z)


print("\nrepetition along existing dimensions with repeat:")
print("the repeat factors specify how often each dimension is tiled")
a = torch.tensor([1, 2, 3])              # shape: [3]
x = a.repeat(2)                          # shape: [6]
f(a, x)

a = torch.tensor([[1, 2], [3, 4]])       # shape: [2, 2]
x = a.repeat(2, 1)                       # shape: [4, 2]
y = a.repeat(1, 3)                       # shape: [2, 6]
z = a.repeat(2, 3)                       # shape: [4, 6]
f(a, x, y, z)

print("\ncombining unsqueeze and repeat for controlled broadcasting:")
print("unsqueeze creates alignment axes, repeat materializes copies")
a = torch.tensor([1., 2., 3.])           # shape: [3]
b = torch.tensor([10., 20.])             # shape: [2]
A = a.unsqueeze(0).repeat(2, 1)          # shape: [2, 3]
B = b.unsqueeze(1).repeat(1, 3)          # shape: [2, 3]
f(a, b, A, B)


print("\nreshape with preserved number of elements:")
print("the total element count must remain invariant")
a = torch.arange(6)                      # shape: [6]
x = a.reshape(2, 3)                      # shape: [2, 3]
f(a, x)

print("\nreshape with an inferred dimension:")
print("exactly one dimension may be set to -1")
a = torch.arange(12)                     # shape: [12]
x = a.reshape(3, -1)                     # shape: [3, 4]
f(a, x)

print("\nflattening and unflattening:")
a = torch.zeros(2, 3, 4)                 # shape: [2, 3, 4]
x = a.reshape(2, -1)                     # shape: [2, 12]
y = a.reshape(-1)                        # shape: [24]
z = x.reshape(2, 3, 4)                   # shape: [2, 3, 4]
f(a, x, y, z)


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