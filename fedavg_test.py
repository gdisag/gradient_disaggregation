import gradient_disaggregation 
from torch.autograd import grad
from torchvision import models, datasets, transforms
import copy
import numpy as np
import numpy as np
import sys
import torch
import torch
import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(1234)

class Net(nn.Module):
    def __init__(self, hidden_size=84):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

nclasses = 10
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

def get_user_datasets(n_users, n_per_user):
    d = []
    dst = datasets.CIFAR10("~/.torch", download=True, transform=transform)
    loader = iter(torch.utils.data.DataLoader(dst, batch_size=1,
                                              shuffle=True))
    k = 0
    for i in range(n_users):
        user_data = []
        for j in range(n_per_user):
            element, label = next(loader)
            element.share_memory_()
            label.share_memory_()
            element, label = copy.deepcopy(element), copy.deepcopy(label)
            user_data.append((element, label))
        d.append(user_data)
    return d        

def get_params(n):
    all_vs = []
    for name, param in n.named_parameters():
        all_vs.append(param.detach().numpy().flatten())
    return np.concatenate(all_vs)    

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_batched_grad_fedavg(user_dataset, local_batchsize, net, epochs=10, momentum=0, lr=1e-2):
    copied_net = copy.deepcopy(net)
    cur_optimizer = optim.SGD(copied_net.parameters(), lr=lr, momentum=momentum)

    user_dataset_indices = np.random.choice(list(range(len(user_dataset))), size=len(user_dataset), replace=False)
    user_dataset = [user_dataset[i] for i in user_dataset_indices]

    for i in range(epochs):
        np.random.shuffle(user_dataset)
        for j in range(0, len(user_dataset), local_batchsize):

            end = min(j+local_batchsize, len(user_dataset))

            inputs = torch.cat([x[0] for x in user_dataset[j:end]], axis=0)
            labels = torch.cat([x[1] for x in user_dataset[j:end]], axis=0)

            cur_optimizer.zero_grad()
            outputs = copied_net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()            
        
            cur_optimizer.step()

    p_net = get_params(net)
    p_copied = get_params(copied_net)
    diff = p_copied - p_net
    
    return diff

def get_batched_grad(user, batchsize, epochs, net, fedavg=True, momentum=0, lr=1e-2):
    return get_batched_grad_fedavg(user, batchsize, net, epochs=epochs, momentum=momentum, lr=lr)

def aggregate_grads(P, user_datasets, batchsize, epochs, momentum=0, lr=1e-2):

    copied_net = copy.deepcopy(net)

    all_grads = []
    gradient = get_params(copied_net).shape[-1]
    for row in range(P.shape[0]):
        print("Aggregating... %d of %d" % (row, P.shape[0]))
        sys.stdout.flush()
        grads = np.zeros((gradient,))
        for col in range(P.shape[1]):
            if P[row,col] == 1:
                batched_grad = get_batched_grad(user_datasets[col], batchsize, epochs, copied_net, momentum=momentum, lr=lr)
                grads += batched_grad
        all_grads.append(grads)
    return np.stack(all_grads)
                
if __name__ == "__main__":
    n_users = 20
    n_rounds = 60
    dataset_size_per_user = 64
    batchsize = 16
    epochs = 4
    granularity = 10

    user_datasets = get_user_datasets(n_users, dataset_size_per_user)
    P = np.random.choice([0, 1], size=(n_rounds, n_users), p=[.8, .2]) 
    G_agg = aggregate_grads(P, user_datasets, batchsize, epochs)
    constraints = gradient_disaggregation.compute_P_constraints(P, granularity) 

    P_star = gradient_disaggregation.reconstruct_participant_matrix(G_agg, constraints, noisy=True, verbose=True)
    
    diff = np.sum(np.abs(P_star-P))
    if diff == 0:
       print("Exactly recovered P!")
    else:
       print("Failed to recover P!")
