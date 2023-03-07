import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
print("module---> my_first_pytorch_model, torch version --->",torch.__version__)


import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

## create a noisy linear correlation
x = np.linspace(0, 1, 1000)
y = 2 * x + np.random.randn(1000) * 0.3

plt.figure(figsize=(10, 8))
plt.scatter(x, y, s=2)
plt.xlabel('x (features)')
plt.ylabel('y (target)')
plt.show()


## Converting the feature & target vectors to tensors
x_tensor = torch.from_numpy(x)
y_tensor =  torch.from_numpy(y)

print("Describing the features...")
print(type(x_tensor))
print(x_tensor.dtype)
print(x_tensor.shape)
print(x_tensor[:10])

print("\n\nDescribing the target...")
print(type(y_tensor))
print(y_tensor.dtype)
print(y_tensor.shape)
print(y_tensor[:10])

# defines the data type of the variable used for w
dtype = torch.float

# defines the device in which we would like to keep the tensor
device = torch.device("cpu")

# States that the weight is a torch tensor. 
# This will allow us to calculate the gradient associated with this variable
# and optimize it using the high level torch modules.
w_tensor = torch.tensor(0.1, device=device, dtype=dtype, requires_grad=True)

# Summary of the weight tensor
print("\n\nDescribing the weights...")
print(type(w_tensor))
print(w_tensor.dtype)
print(w_tensor.shape)
print(w_tensor)


# defines the figure to plot
fig, ax = plt.subplots(figsize=(10, 8))
plt.xlabel('x (features)')
plt.ylabel('y (target)')

# plot the original data
line1 = ax.scatter(x, y, s=2)

# estimates the predicted Y and plots a new line
reg_x = np.linspace(0,1,100)
reg_yhat = np.linspace(0,1,100) * w_tensor.detach().numpy()
line2, = ax.plot(reg_x, reg_yhat, 'black')

# display legend
plt.gca().legend(['predicted reg line', 'data'])

plt.show()



# defines the MSE loss
loss_func = torch.nn.MSELoss(reduction='sum')

print("Summary of loss function...")
display.display(loss_func)


# defines the SGD optimizer
learning_rate = 1e-4
optimizer = torch.optim.SGD([w_tensor], lr=learning_rate)

print("\n\nSummary of optimizer...")
display.display(optimizer)

print("\n\n List of parameters to optimize...")
display.display(optimizer.param_groups)