import torch
import torchvision
from torch import nn, optim
from torch.utils.data import DataLoader

from metrics_recorder import MetricsRecorder
from simple_cnn import SimpleCnn
from simple_mlp import SimpleMlp
import torchvision.transforms as transforms


def train(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    batch_size = 64

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    mlp_model = SimpleMlp()
    cnn_model = SimpleCnn()

    criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

    num_epochs = 10

    mr = MetricsRecorder().start()
    # for epoch in range(num_epochs):
    #     train_loss, train_acc = train(mlp_model, train_loader, criterion, mlp_optimizer)
    #     test_loss, test_acc = evaluate(mlp_model, test_loader, criterion)
    #     print(f"Epoch: {epoch}, MLP Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    # duration = mr.end().get_metrics()
    # print(f'MLP duration: {duration}')

    mr.start()
    for epoch in range(num_epochs):
        train_loss, train_acc = train(cnn_model, train_loader, criterion, cnn_optimizer)
        test_loss, test_acc = evaluate(cnn_model, test_loader, criterion)
        print(f"Epoch: {epoch}, CNN Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    duration = mr.end().get_metrics()
    print(f'CNN duration: {duration}')

    # mlp_test_loss, mlp_test_acc = evaluate(mlp_model, test_loader, criterion)
    cnn_test_loss, cnn_test_acc = evaluate(cnn_model, test_loader, criterion)

    # print(f"MLP Test Loss: {mlp_test_loss:.4f}, Test Accuracy: {mlp_test_acc:.2f}%")
    print(f"CNN Test Loss: {cnn_test_loss:.4f}, Test Accuracy: {cnn_test_acc:.2f}%")
