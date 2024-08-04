# Import necessary modules
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

import sys
import select



# Define ResidualBlock class
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

# Define ResNet_CNN class
class ResNet_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet_CNN, self).__init__()

        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, padding=1)   ,
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ResidualBlock(in_channels=128, out_channels=128)  ,
            ResidualBlock(in_channels=128, out_channels=64,use_1x1conv=True,strides=2)  ,
            ResidualBlock(in_channels=64, out_channels=64)  ,
            ResidualBlock(in_channels=64, out_channels=32,use_1x1conv=True,strides=2)  ,
            ResidualBlock(in_channels=32, out_channels=32)  ,
            ResidualBlock(in_channels=32, out_channels=16,use_1x1conv=True,strides=2)  ,
            ResidualBlock(in_channels=16, out_channels=16)  ,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=16, out_features=num_classes) 
        )
                                                                                        
    def forward(self, x):
        x = self.sequential(x)
        return x

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = ResNet_CNN().to(device)
# model = torch.jit.load("traced_model.pt").to(device)
model.load_state_dict(torch.load("model_state.pt"))

print(model)
X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32).to(device)
for name, layer in model.named_children():
    print(name)
    if isinstance(layer, nn.Sequential):
        for name1, layer1 in layer.named_children():
            X = layer1(X)
            print("  ", name1, 'output shape:', X.shape)
    else:
        X = layer(X)
        print("  ", name, 'output shape:', X.shape)


# Define loss function
loss_f = nn.CrossEntropyLoss()

def is_key_pressed():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

def wait_until_save():
    if is_key_pressed():
        key = sys.stdin.read(1)
        if key == 'q':
            #Save Param
            torch.save(model.state_dict(), "model_state.pt")
            # 保存模型
            example_input = torch.rand(1, 1, 28, 28).to(device)
            traced_model = torch.jit.trace(model, example_input)
            torch.jit.save(traced_model, "traced_model.pt")
            print("Model saved.")
            sys.exit(0)  # 退出程序

# Define optimizer
learning_rate = 0.000002
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

log_interval = 10

# Training function
def train(epochs):
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_f(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch:{} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()
                ))
            wait_until_save()
        evaluate_train_set()
        test()
        print('\n')

# Test function
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_f(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.8f}, Accuracy: {}/{} ({:.3f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))

def evaluate_train_set():
    model.eval()
    train_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            train_loss += loss_f(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    train_loss /= len(train_loader.dataset)
    print('Train set: Average loss: {:.8f}, Accuracy: {}/{} ({:.3f}%)'.format(
        train_loss, correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)
    ))

# Training and testing
if __name__ == '__main__':
    epochs = 20
    batch_size_train = 1000
    batch_size_test = 1000
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True
    )


    random_seed = 1
    torch.manual_seed(random_seed)
    train(epochs)
    # evaluate_train_set()
    # test()

#Save Param
torch.save(model.state_dict(), "model_state.pt")

# Save the model
example_input = torch.rand(1, 1, 28, 28).to(device)
traced_model = torch.jit.trace(model, example_input)
torch.jit.save(traced_model, "traced_model.pt")