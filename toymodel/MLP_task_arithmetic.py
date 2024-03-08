import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import copy
import wandb
import argparse

# Check if CUDA is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layers(x.view(-1, 784))

def train_model(model, train_loader, criterion, optimizer, num_epochs=1, freeze_layers=True):
    if freeze_layers:
        # Freeze all layers except the last one
        for param in model.layers[:-1].parameters():
            param.requires_grad = False
    else:
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True

    model.train()
    for epoch in range(num_epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        train_acc = evaluate_model(model, train_loader)
        print(f"Epoch{epoch} Acc={train_acc}")

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = correct / len(test_loader.dataset)
    return accuracy

def main():
    # Initialize models, loss, and optimizer
    model_0 = SimpleMLP().to(device)
    model_1 = copy.deepcopy(model_0)
    model_2 = copy.deepcopy(model_0)
    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(model_1.parameters())
    optimizer_2 = optim.Adam(model_2.parameters())

    # Perform Linear Probing
    print("Starting Linear Probing")
    train_model(model_1, train_loader_1, criterion, optimizer_1, num_epochs=args.linear_probing_epochs, freeze_layers=True)
    train_model(model_2, train_loader_2, criterion, optimizer_2, num_epochs=args.linear_probing_epochs, freeze_layers=True)

    # Evaluate models after Linear Probing
    acc_lp_1 = evaluate_model(model_1, test_loader)
    acc_lp_2 = evaluate_model(model_2, test_loader)
    wandb.log({"Linear Probing Accuracy Model 1": acc_lp_1, "Linear Probing Accuracy Model 2": acc_lp_2})

    # Train models with all layers
    print("Starting Full Training")
    train_model(model_1, train_loader_1, criterion, optimizer_1, num_epochs=args.epochs, freeze_layers=False)
    train_model(model_2, train_loader_2, criterion, optimizer_2, num_epochs=args.epochs, freeze_layers=False)

    # Evaluate models
    acc_1 = evaluate_model(model_1, test_loader)
    acc_2 = evaluate_model(model_2, test_loader)
    wandb.log({"Full Training Accuracy Model 1": acc_1, "Full Training Accuracy Model 2": acc_2})

    # Perform Task Arithmetic
    initial_state_dict = model_0.state_dict()
    model_1_diff = {name: model_1.state_dict()[name] - initial_state_dict[name] for name in initial_state_dict}
    model_2_diff = {name: model_2.state_dict()[name] - initial_state_dict[name] for name in initial_state_dict}
    combined_weights = {name: initial_state_dict[name] + model_1_diff[name] + model_2_diff[name] for name in initial_state_dict}

    new_model = SimpleMLP().to(device)
    new_model.load_state_dict(combined_weights)

    # Evaluate new model
    acc_combined = evaluate_model(new_model, test_loader)
    wandb.log({"Combined Model Accuracy": acc_combined})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a 3-layer MLP for classification.')
    parser.add_argument('--epochs', type=int, default=10, help='LP')
    parser.add_argument('--linear_probing_epochs', type=int, default=5, help='LP')
    args = parser.parse_args()

    # Initialize WandB
    wandb.init(project="task_arithmetic_with_mlp")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    train_dataset_1 = torch.utils.data.Subset(full_dataset, [i for i in range(len(full_dataset)) if full_dataset.targets[i] < 5])
    train_dataset_2 = torch.utils.data.Subset(full_dataset, [i for i in range(len(full_dataset)) if full_dataset.targets[i] >= 5])
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader_1 = torch.utils.data.DataLoader(train_dataset_1, batch_size=1024, shuffle=True)
    train_loader_2 = torch.utils.data.DataLoader(train_dataset_2, batch_size=1024, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False)

    main()
