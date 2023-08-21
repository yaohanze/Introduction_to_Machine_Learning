#!/usr/bin/env python

"""
    pitfalls.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Data generating process
class DataGen():
    def __init__(self, d):
        self.v_star = np.random.randint(2, size=d)
        self.d = d

    def get_batch(self, N):
        X = np.random.randint(2, size=(N,self.d))
        count = X.dot(self.v_star)
        Y = count % 2 # determines even (0) or odd (1)
        Y = -2*Y+1 # rescaling
        return torch.from_numpy(X).float(),torch.from_numpy(Y).long()

# Implement your neural net here
# If you are not familiar with pytorch, you may want to  
# take a look at the basic tutorial here 
# https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
class Net(nn.Module):
    def __init__(self, d):
        super(Net, self).__init__()
        # TO IMPLEMENT Part (a)
        # here you should declare functions for layers with parameters
        self.fc1 = nn.Linear(d, 10*d, bias=True)
        self.fc2 = nn.Linear(10*d, 1, bias=True)

    def forward(self, x):
        # TO IMPLEMENT Part (a)
        # here you should trace the forward computation of the network
        # using functions declared in __init__
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.float()

def set_parameters(d, w, b, alpha, beta, net):
    '''
        This function assigns parameters in the network based on matching shape
    '''

    # First, we reshape parameters
    w = w.reshape(10*d,d)
    b = b.reshape(10*d,)
    alpha = alpha.reshape(1,10*d)
    beta = beta.reshape(1,)

    # Next, we loop through all parameters in the network
    set_w = False; set_b = False; set_alpha = False; set_beta = False
    for p in net.parameters():
        if p.shape == w.shape and not set_w:
            p.data = torch.from_numpy(w).type(torch.float)
            set_w = True
        elif p.shape == b.shape and not set_b:
            p.data = torch.from_numpy(b).type(torch.float)
            set_b = True
        elif p.shape == alpha.shape and not set_alpha:
            p.data = torch.from_numpy(alpha).type(torch.float)
            set_alpha = True
        elif p.shape == beta.shape and not set_beta:
            p.data = torch.from_numpy(beta).type(torch.float)
            set_beta = True
        else:
            print("Unexpected parameter list for network!")
            print("Expected one of:", w.shape, b.shape, alpha.shape, beta.shape)
            print("Got:",p.shape)
            assert False
"""
# Initializing the data generation process
d = 5
data = DataGen(d)

### Defining network parameters by hand

# Initializing the network
net = Net(d)

# Defining and setting parameters for perfect approximation
# TO IMPLEMENT Part (a)
v = data.v_star
w = np.zeros((10*d, d))
for i in range(10*d):
    if i <= 3*d/2-1:
        w[i,:] = v
b = np.zeros((10*d, 1))
alpha = np.zeros((1, 10*d))
for j in range((d//2)+1):
    b[3*j,0] = -(2*j-0.5)
    b[3*j+1,0] = -2*j
    b[3*j+2,0] = -(2*j+0.5)
    alpha[0,3*j] = 4
    alpha[0,3*j+1] = -8
    alpha[0,3*j+2] = 4
beta = np.array([-1])
set_parameters(d, w, b, alpha, beta, net)

# Checking that network perfectly approximates 10,000 examples
inputs, labels = data.get_batch(10000)
assert sum(net(inputs).reshape(-1) - labels.float()) == 0

### Learning network parameters

# Initializing the network and data generation
net = Net(d)
data = DataGen(d)
"""
# Defining the loss
def myLoss(outputs, labels):
    # TO IMPLEMENT Part (b)
    # should return 1/b * sum_{i=1}^b loss(f_p(x_i),y_i)
    outputs = outputs.view(-1,1)
    labels = labels.view(-1,1)
    n = outputs.size()[0]
    ones = torch.ones(n, 1, dtype=float, requires_grad=True)
    arg = torch.mul(outputs, labels)
    arg = torch.mul(arg, -1)
    arg = torch.add(arg, 1)
    l = F.relu(arg)
    ml = torch.sum(l)
    ml = torch.div(ml, n)
    return ml
"""
loss_function = myLoss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
batch_size = 100
num_iter = 5*10**4
loss_over_iteration = []

# we don't need to loop over epochs because we consider an infinite data source
for i in range (num_iter): 
    # get the inputs
    inputs, labels = data.get_batch(batch_size)

    # zero the parameter gradients
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = loss_function(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    loss_over_iteration.append(loss.item())
    if i % 200 == 0:    # print every 200 mini-batches
        print('[%5d] loss: %.3f' %
              (i, loss.item()))
print('Finished Training')
"""
# Plotting the training curve
# TO IMPLEMENT Part (b)
d_list = [5, 10, 30]
for d1 in d_list:
    net1 = Net(d1)
    data1 = DataGen(d1)
    loss_function1 = myLoss
    optimizer1 = optim.SGD(net1.parameters(), lr=0.001, momentum=0.9)
    batch_size1 = 100
    num_iter1 = 5*10**4
    loss_over_iteration1 = []
    for m in range(num_iter1):
        inputs1, labels1 = data1.get_batch(batch_size1)
        optimizer1.zero_grad()
        outputs1 = net1(inputs1)
        loss1 = loss_function1(outputs1, labels1)
        loss1.backward()
        optimizer1.step()
        loss_over_iteration1.append(loss1.item())
        if m%200 == 0:
            print('[%5d] loss: %.3f' % (m, loss1.item()))
    print('Finished Training d=%2d' % d1)
    plt.plot(np.array(loss_over_iteration1))
plt.legend(('d=5', 'd=10', 'd=30'))
plt.xlabel('Number of iteration')
plt.ylabel('Loss')
plt.show()
