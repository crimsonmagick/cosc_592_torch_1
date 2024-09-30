import time

import torch
import torch_directml
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

# device = torch_directml.device() if torch_directml.is_available() else torch.device("cpu")
device = torch.device("cpu")
# device = torch_directml.device()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


input_size = 28 * 28
hidden_size = 128
num_classes = 10
model = MLP(input_size, hidden_size, num_classes).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 10
before_all = time.time()
for epoch in range(num_epochs):
    before_epoch = time.time()
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_predictions.extend(predicted.tolist())
            all_labels.extend(labels.tolist())

        accuracy = 100 * correct / total
        after_epoch = time.time()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {accuracy:.2f}%, Epoch "
          f"Execution Time: {after_epoch - before_epoch} s")
after = time.time()

print('Final Evaluation')
print(f'Accuracy: {accuracy_score(all_labels, all_predictions) * 100:.2f}%')
print(f'{device} - {optimizer} Execution time: {after - before_all} s')
