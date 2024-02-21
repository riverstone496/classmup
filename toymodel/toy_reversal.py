import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np

class ThreeLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_type):
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU() if activation_type == 'ReLU' else nn.Identity()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, optimizer, data_loader, epochs, train_loss=None):
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for x, y in data_loader:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.long())
            loss.backward()
            optimizer.step()
        if epoch%10 == 0 or epoch == epochs-1:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
        if train_loss is not None and train_loss > loss:
            print(f'Epoch {epoch+1}, Loss: {loss.item()}')
            break
    return model

def evaluate_loss(model, data_loader):
    model.eval()  # Set the model to evaluation mode
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            output = model(x)
            loss = criterion(output, y.long())
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    average_loss = total_loss / len(data_loader)
    accuracy = 100 * correct / total
    print(f'Test Loss: {average_loss}, Accuracy: {accuracy}%')
    model.train()  # Set the model back to training mode


def generate_data(num_samples, input_size, output_size, teacher_model, shift_class = False):
    x = torch.randn(num_samples, input_size)
    y_raw = teacher_model(x)
    y = torch.argmax(y_raw, dim=1)  # Convert to class labels
    if shift_class:
        y = (y + 1) % output_size  # Shift class labels by one
    return x, y

def shift_rows(tensor):
    last_column = tensor[:, -1].unsqueeze(1)
    rest_columns = tensor[:, :-1]
    shifted_tensor = torch.cat((last_column, rest_columns), dim=1)
    return shifted_tensor

def main():
    parser = argparse.ArgumentParser(description='Train a 3-layer MLP for classification.')
    parser.add_argument('--width', type=int, default=2048, help='Width of the hidden layers.')
    parser.add_argument('--input_size', type=int, default=32, help='Input size.')
    parser.add_argument('--output_size', type=int, default=4, help='Number of classes for classification.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--lp_epochs', type=int, default=0, help='Number of epochs for training.')

    parser.add_argument('--train_loss_limit', type=float, default=1e-5, help='Number of epochs for training.')
    parser.add_argument('--ft_max_epochs', type=int, default=10000, help='Number of epochs for fine-tuning.')
    
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate.')
    parser.add_argument('--finetune_lr', type=float, default=1, help='Learning rate for fine-tuning.')
    parser.add_argument('--lp_lr', type=float, default=1e-1, help='Learning rate for fine-tuning.')
    parser.add_argument('--finetune_lr_muP', type=float, default=1e-3, help='Fine-tuning learning rate for muP parametrization.')
    parser.add_argument('--num_samples', type=int, default=2048, help='Number of samples for initial training.')
    parser.add_argument('--num_finetuning_samples', type=int, default=30, help='Number of samples for fine-tuning.')
    parser.add_argument('--parametrization', choices=['muP', 'NTK', 'SP', 'LP'], default='NTK', help='Type of parametrization.')
    parser.add_argument('--activation', choices=['ReLU', 'Linear'], default='Linear', help='Activation function.')
    args = parser.parse_args()

    np.random.seed(234)
    torch.manual_seed(234)

    model = ThreeLayerMLP(args.input_size, args.width, args.output_size, args.activation)
    teacher_model = ThreeLayerMLP(args.input_size, args.width, args.output_size, args.activation)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    x_train, y_train = generate_data(args.num_samples, args.input_size, args.output_size, teacher_model)
    train_data_loader = [(x_train, y_train)]
    model = train(model, optimizer, train_data_loader, args.epochs)
    
    x_test, y_test = generate_data(1000, args.input_size, args.output_size, teacher_model)
    test_data_loader = [(x_test, y_test)]
    evaluate_loss(model, test_data_loader)

    nn.init.kaiming_normal_(model.fc3.weight)

    # Fine-tuning Optimizer
    if args.parametrization == 'muP':
        finetune_optimizer = optim.SGD([
            {'params': model.fc1.parameters(), 'lr': args.finetune_lr_muP * args.width},
            {'params': model.fc2.parameters(), 'lr': args.finetune_lr_muP},
            {'params': model.fc3.parameters(), 'lr': args.finetune_lr_muP / args.width}
        ], lr=args.finetune_lr_muP, momentum=0.9)
    elif args.parametrization == 'NTK':
        finetune_optimizer = optim.SGD([
            {'params': model.fc1.parameters(), 'lr': args.finetune_lr},
            {'params': model.fc2.parameters(), 'lr': args.finetune_lr / args.width},
            {'params': model.fc3.parameters(), 'lr': args.finetune_lr / args.width}
        ], lr=args.finetune_lr, momentum=0.9)
    elif args.parametrization == 'LP':
        finetune_optimizer = optim.SGD([
            {'params': model.fc1.parameters(), 'lr': 0},
            {'params': model.fc2.parameters(), 'lr': 0},
            {'params': model.fc3.parameters(), 'lr': args.lp_lr}
        ], lr=args.lp_lr, momentum=0.9)
    else:
        finetune_optimizer = optim.SGD(model.parameters(), lr=args.finetune_lr / args.width, momentum=0.9)

    x_finetune, y_finetune = generate_data(args.num_finetuning_samples, args.input_size, args.output_size, teacher_model, True)
    #x_finetune = shift_rows(x_finetune)
    finetune_data_loader = [(x_finetune, y_finetune)]

    if args.lp_epochs > 0:
        lp_optimizer = optim.SGD([
            {'params': model.fc1.parameters(), 'lr': 0},
            {'params': model.fc2.parameters(), 'lr': 0},
            {'params': model.fc3.parameters(), 'lr': args.lp_lr}
        ], lr=args.lp_lr, momentum=0.9)
        model = train(model, lp_optimizer, finetune_data_loader, args.lp_epochs)

    model = train(model, finetune_optimizer, finetune_data_loader, args.ft_max_epochs, args.train_loss_limit)

    x_test, y_test = generate_data(1000, args.input_size, args.output_size, teacher_model, True)
    #x_test = shift_rows(x_test)
    test_data_loader = [(x_test, y_test)]
    evaluate_loss(model, test_data_loader)

if __name__ == '__main__':
    main()
