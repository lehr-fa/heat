import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms

import heat as ht


# global batch size
bGlo = 32

# local batch size
bLoc = 0


# defining a simple fc nn with 2 hidden layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lin1 = nn.Linear(784, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 10)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=1)
        return x


# hook function for blocking gradient data exchange
def blocking_hook(grad_loc):
    # Pytorch Doc says, :attr:`grad` may not be modified itself, so it has to be cloned
    # (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
    # Seems to be true, since otherwise a Runtime Error is thrown when working on it
    grad_loc_cpy = grad_loc.clone()

    # counterbalance local gradient averaging
    grad_loc_cpy *= bLoc

    # wrap local gradient into heat tensor
    grad_ht = ht.array(grad_loc_cpy, copy=False)

    # perform MPI Allreduce to compute global gradient
    grad_ht.comm.Allreduce(ht.MPI.IN_PLACE, grad_ht, ht.MPI.SUM)

    # unwrap global gradient from heat tensor
    grad_glo = grad_ht._DNDarray__array

    # global gradient averaging
    grad_glo /= bGlo

    return grad_glo


# hook function for non-blocking gradient data exchange
def nonblocking_hook(grad_loc):
    # Pytorch Doc says, :attr:`grad` may not be modified itself, so it has to be cloned
    # (cf. https://pytorch.org/docs/stable/tensors.html#torch.Tensor.register_hook).
    # Seems to be true, since otherwise a Runtime Error is thrown when working on it
    grad_loc_cpy = grad_loc.clone()

    # counterbalance local gradient averaging
    grad_loc_cpy *= bLoc

    # wrap local gradient into heat tensor
    grad_ht = ht.array(grad_loc_cpy, copy=False)

    # perform MPI IAllreduce to compute global gradient, returns wait handle
    wait_handle = grad_ht.comm.Iallreduce(ht.MPI.IN_PLACE, grad_ht, ht.MPI.SUM)

    # inject wait handle into local gradient
    setattr(grad_loc_cpy, "wait_handle", wait_handle)

    return grad_loc_cpy


net = Net()

# registering hooks for all model parameter tensors (here: one bias and one weights tensor per layer)
for param in net.parameters():
    param.register_hook(blocking_hook)


criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)


# take MNIST dataset
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=bGlo,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=bGlo,
    shuffle=True,
)


# training phase
net.train()
for epoch in range(2):
    for batch_idx, (data, target) in enumerate(train_loader):
        input_ = data.view(data.size(0), -1)
        bLoc = input_.size()[0]
        optimizer.zero_grad()
        output = net(input_)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


# testing phase
net.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        input_ = data.view(data.size(0), -1)
        output = net(input_)
        test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

test_loss /= len(test_loader.dataset)

print(
    "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
        test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
    )
)
