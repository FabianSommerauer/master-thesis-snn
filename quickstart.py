import numpy as np
from torch import Tensor
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import norse.torch as norse
import matplotlib.pyplot as plt

#import norse.task.cifar10

#plt.style.use("../resources/matplotlibrc")

batch_size = 128
data_path = '/tmp/data/mnist'

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

_ = torch.manual_seed(0)


class ToBinaryTransform(object):
    def __init__(self, thresh):
        self.thresh = thresh

    def __call__(self, data: Tensor):
        return (data > self.thresh).to(data.dtype)


class MaxPotentialDecode(nn.Module):
    def __init__(self):
        super(MaxPotentialDecode, self).__init__()

    def forward(self, membrane_potential: Tensor) -> Tensor:
        return membrane_potential.max(dim=0).values

class RateDecode(nn.Module):
    def __init__(self):
        super(RateDecode, self).__init__()

    def forward(self, spk_trns: Tensor) -> Tensor:
        return spk_trns.mean(dim=0)


transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,)),
    # ToBinaryTransform(0.5)
])

mnist_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

beta = 0.9

net = norse.SequentialState(
    nn.Conv2d(1, 8, 5),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 5),
    nn.MaxPool2d(2),
    #nn.Sigmoid(),
    nn.Flatten(),
    #norse.ConstantCurrentLIFEncoder(32, dt=0.001),
    #norse.SpikeLatencyLIFEncoder(32, dt=0.001),
    #norse.PoissonEncoder(32),  # (time, batch, features)
    nn.Linear(16 * 4 * 4, 10),
    #nn.Tanh(),
    #norse.LIF(),
    #nn.Linear(32, 10),
    #norse.LI(),
    #MaxPotentialDecode(),
).to(device)


def forward_pass(net, data):
    return net(data)


output, state = forward_pass(net, mnist_train[0][0][None, ...].to(device))
output


optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, betas=(0.9, 0.999))
loss_fn = nn.CrossEntropyLoss()

num_epochs = 1  # run for 1 epoch - each data sample is seen only once
num_steps = 25  # run for 25 time steps

loss_hist = []  # record loss over iterations
acc_hist = []  # record accuracy over iterations

# training loop
for epoch in range(num_epochs):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)

        net.train()
        output, state = forward_pass(net, data)
        loss_val = loss_fn(output, targets)
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        loss_hist.append(loss_val.item())

        if i % 25 == 0:
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")

            # check accuracy on a single batch
            acc = np.mean(np.argmax(output.detach().cpu().numpy(), axis=1) == targets.cpu().numpy())
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")


total_acc = 0
for i, (data, targets) in enumerate(iter(test_loader)):
    data = data.to(device)
    targets = targets.to(device)

    net.eval()
    output, state = forward_pass(net, data)
    total_acc += np.mean(np.argmax(output.detach().cpu().numpy(), axis=1) == targets.cpu().numpy())

print(f"Test Accuracy: {total_acc / len(test_loader) * 100:.2f}%")