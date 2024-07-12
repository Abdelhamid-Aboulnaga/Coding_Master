import torch
import math

class LegPoly(torch.autograd.Function):
    
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * 1.5 * (5 * input **2 - 1)
dtype = torch.float32
torch.set_default_device("cpu")
steps = 2000
x = torch.linspace(-math.pi, math.pi, steps, dtype=dtype)
y = torch.sin(x)
a,b,c,d = [torch.full((), s, dtype=dtype, requires_grad=True) for s in [0.0, -1.0, 0.0, 0.3]]
weights = [a,b,c,d]
learning_rate = 5e-6
for i in range(steps):
    P3 = LegPoly.apply
    y_pred = a + b * P3(c + d * x)
    loss = (y_pred - y).pow(2).sum()
    if i%100 == 99:
        print(i, loss.item())
    loss.backward()
    with torch.no_grad():
        for w in weights:
            w -= learning_rate * w.grad
            w.grad = None
print(f'fitting function when using P3 is y = {a.item()} + {b.item()} * P3({c.item()} + {d.item()} x)')