import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import argparse

class ThreeLayerMLP(nn.Module):
    """
    A three-layer Multilayer Perceptron (MLP) with customizable activation function.
    """
    def __init__(self, input_size, hidden_size, output_size, activation_type):
        """
        Initializes the MLP model with three linear layers.
        :param input_size: size of the input features
        :param hidden_size: size of the hidden layers
        :param output_size: size of the output layer
        :param activation_type: type of activation function ('ReLU' or 'Linear')
        """
        super(ThreeLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
        self.fc2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc3 = nn.Linear(hidden_size, output_size, bias=False)
        self.activation = nn.ReLU() if activation_type == 'ReLU' else nn.Identity()

    def forward(self, x):
        """
        Forward pass through the MLP.
        :param x: input tensor
        :return: output tensor
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

class TransferModel(nn.Module):
    """
    A model for transfer learning based on an existing base model with a new output layer.
    """
    def __init__(self, base_model, new_output_size):
        """
        Initializes the transfer model by reusing layers from the base model and adding a new output layer.
        :param base_model: the original model from which to transfer
        :param new_output_size: size of the new output layer
        """
        super(TransferModel, self).__init__()
        self.fc1 = base_model.fc1
        self.fc2 = base_model.fc2
        self.fc3 = nn.Linear(base_model.fc2.out_features, new_output_size, bias=False)
        nn.init.kaiming_normal_(self.fc3.weight)
        self.activation = base_model.activation

    def forward(self, x):
        """
        Forward pass through the transfer model.
        :param x: input tensor
        :return: output tensor
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

def register_fhook(model: torch.nn.Module):
    # forward hook function
    def forward_hook(_module, in_data, out_data):
        _module.out_data = out_data.detach().clone()
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        module.register_forward_hook(forward_hook)
    return model

def fetch_h(model, inputs_for_dh):
    model = register_fhook(model)
    y = model(inputs_for_dh)
    pre_act_dict = {}
    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
              continue
        if all(not p.requires_grad for p in module.parameters()):
            continue
        pre_act_dict[name] = module.out_data
    return pre_act_dict

def log_h_delta(epoch, model, init_pre_act_dict, tmp_pre_act_dict, inputs_for_dh, prefix = '', tmp_dh=True):
    pre_act_dict = fetch_h(model, inputs_for_dh)
    h_norm_dict = {}
    dh_norm_dict = {}
    int_dh_norm_dict = {}
    for mname in pre_act_dict.keys():
        h_norm_dict[mname] = torch.abs(pre_act_dict[mname]).mean(dtype=torch.float32).item()
        dh_norm_dict[mname] = torch.abs(pre_act_dict[mname] - init_pre_act_dict[mname]).mean(dtype=torch.float32).item()
        int_dh_norm_dict[mname] = torch.abs(pre_act_dict[mname] - tmp_pre_act_dict[mname]).mean(dtype=torch.float32).item()
    log = {'epoch': epoch,
           prefix + 'h/':h_norm_dict,
           prefix + 'dh/': dh_norm_dict,}
    if tmp_dh:
        log[prefix + 'tmp_dh/'] = int_dh_norm_dict
        tmp_pre_act_dict = fetch_h(model, inputs_for_dh=inputs_for_dh)
    if args.wandb:
        wandb.log(log)
    return tmp_pre_act_dict

def prepare_log_h_delta(model, test_loader):
    for i, data in enumerate(test_loader, 0):
        inputs, labels = data
        break
    init_pre_act_dict = fetch_h(model, inputs)
    tmp_pre_act_dict  = fetch_h(model, inputs)
    return inputs, init_pre_act_dict, tmp_pre_act_dict

def initialize_optimizer(model, learning_rate, momentum, parametrization, width, muP_factor = 1e-2):
    """
    Initializes the optimizer with custom learning rates for fine-tuning.
    :param model: the model to optimize
    :param learning_rate: base learning rate
    :param momentum: momentum for SGD optimizer
    :param parametrization: type of parametrization ('muP', 'NTK', 'LP')
    :param width: width of the hidden layers, affects learning rates in 'muP' parametrization
    :return: initialized optimizer
    """
    if parametrization == 'muP':
        return optim.SGD([
            {'params': model.fc1.parameters(), 'lr': muP_factor * learning_rate * width},
            {'params': model.fc2.parameters(), 'lr': muP_factor * learning_rate},
            {'params': model.fc3.parameters(), 'lr': muP_factor * learning_rate / width}
        ], lr=learning_rate, momentum=momentum)
    elif parametrization == 'NTK':
        return optim.SGD([
            {'params': model.fc1.parameters(), 'lr': learning_rate},
            {'params': model.fc2.parameters(), 'lr': learning_rate / width},
            {'params': model.fc3.parameters(), 'lr': learning_rate / width}
        ], lr=learning_rate, momentum=momentum)
    elif parametrization == 'LP':
        return optim.SGD([
            {'params': model.fc1.parameters(), 'lr': 0},
            {'params': model.fc2.parameters(), 'lr': 0},
            {'params': model.fc3.parameters(), 'lr': learning_rate}
        ], lr=learning_rate, momentum=0.0)
    else:
        return optim.SGD(model.parameters(), lr=learning_rate / width, momentum=momentum)

def test(model, criterion, test_data_loader):
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        total_loss, correct, total = 0, 0, 0
        for x, y in test_data_loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y.long())
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        test_average_loss = total_loss / len(test_data_loader)
        test_accuracy = 100 * correct / total
    return test_average_loss, test_accuracy

def train(model, optimizer, criterion, data_loader, test_data_loader, epochs, early_stopping_loss=None, prefix = ''):
    """
    Trains and evaluates the model.
    :param model: the model to train
    :param optimizer: the optimizer
    :param criterion: loss function
    :param data_loader: training data loader
    :param test_data_loader: test data loader
    :param epochs: number of training epochs
    :param early_stopping_loss: early stopping threshold for loss
    :return: trained model
    """

    # Prepare for logging dh
    inputs, init_pre_act_dict, tmp_pre_act_dict = prepare_log_h_delta(model, test_data_loader)

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y.long())
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(data_loader)
        accuracy = 100 * correct / total
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            test_loss, test_acc = test(model, criterion, test_data_loader)
            tmp_pre_act_dict = log_h_delta(epoch, model, init_pre_act_dict, tmp_pre_act_dict, inputs_for_dh=inputs, prefix = prefix)
            print(f'Epoch {epoch+1}, Train Loss: {average_loss}, Train Acc: {accuracy}%, Test Loss: {test_loss}, Test Acc: {test_acc}')
            if args.wandb:
                wandb.log({
                    'epoch' : epoch+1,
                    prefix + 'Train Loss' : average_loss,
                    prefix + 'Train Acc' : accuracy,
                    prefix + 'Test Loss' : test_loss,
                    prefix + 'Test Acc' : test_acc,
                })

        if early_stopping_loss is not None and average_loss <= early_stopping_loss:
            test_loss, test_acc = test(model, criterion, test_data_loader)
            print(f'Early stopping at epoch {epoch+1}!')
            print(f'Epoch {epoch+1}, Train Loss: {average_loss}, Train Acc: {accuracy}%, Test Loss: {test_loss}, Test Acc: {test_acc}')
            tmp_pre_act_dict = log_h_delta(epoch, model, init_pre_act_dict, tmp_pre_act_dict, inputs_for_dh=inputs, prefix = prefix, tmp_dh = False)
            break
    return model

def generate_data(num_samples, input_size, output_size, teacher_model, shift_class=False):
    """
    Generates synthetic data using a teacher model.
    :param num_samples: number of samples to generate
    :param input_size: input size
    :param output_size: output size
    :param teacher_model: model used to generate labels
    :param shift_class: whether to shift class labels by one
    :return: tuple of input tensor and labels
    """
    x = torch.randn(num_samples, input_size)
    y_raw = teacher_model(x)
    y = torch.argmax(y_raw, dim=1)  # Convert to class labels
    if shift_class:
        y = (y + 1) % output_size  # Shift class labels by one
    return x, y

def setup_data_loaders(num_samples, input_size, output_size, teacher_model, shift_class = False):
    """
    Sets up data loaders for training and testing.
    :param num_samples: Number of samples for training data
    :param input_size: The size of the input features
    :param output_size: The size of the output classes
    :param teacher_model: The model used to generate synthetic labels
    :return: A tuple of (train_data_loader, test_data_loader)
    """
    x_train, y_train = generate_data(num_samples, input_size, output_size, teacher_model, shift_class)
    train_data_loader = [(x_train, y_train)]
    x_test, y_test = generate_data(1000, input_size, output_size, teacher_model, shift_class)
    test_data_loader = [(x_test, y_test)]
    return train_data_loader, test_data_loader

def model_shift(module, cor):
    a_sqrt = torch.sqrt(torch.tensor(cor))
    W = module.weight.data
    J = torch.randn_like(W)
    new_weights = a_sqrt * W + (1 - a_sqrt) * J
    module.weight.data = new_weights
    return module

def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.wandb:
        config = vars(args).copy()
        wandb.init( config=config,
                    entity=os.environ.get('WANDB_ENTITY', None),
                    project=os.environ.get('WANDB_PROJECT', None),
                    )

    # Initialize models and optimizer
    model = ThreeLayerMLP(args.input_size, args.width, args.output_size, args.activation)
    teacher_model = ThreeLayerMLP(args.input_size, args.width, args.output_size, args.activation)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    model = model.to(device)

    # Setup data loaders
    train_data_loader, test_data_loader = setup_data_loaders(args.num_samples, args.input_size, args.output_size, teacher_model)

    # Determine loss function based on argument
    criterion = nn.CrossEntropyLoss() if args.loss_type == 'CE' else nn.MSELoss()

    # Train model
    model = train(model, optimizer, criterion, train_data_loader, test_data_loader, args.epochs, args.train_loss_limit, 'PreTraining/')

    # Fine-tuning phase (if required)
    print("\nStart Finetuning\n")
    if args.finetune_lr > 0:
        # Create Transfer model
        model = TransferModel(model,new_output_size=args.output_size)
        # Create Teacher Model
        model.fc1 = model_shift(model.fc1, args.input_shift_cor)
        model.fc3 = model_shift(model.fc3, args.output_shift_cor)

        # Adjust the optimizer for fine-tuning
        finetune_optimizer = initialize_optimizer(model, args.finetune_lr, 0.9, args.parametrization, args.width)
        # Generate new data for fine-tuning
        finetune_data_loader, test_data_loader = setup_data_loaders(args.fine_tuning_num_samples, args.input_size, args.output_size, teacher_model, shift_class=args.class_shift)
        # Fine-tune model
        model = train(model, finetune_optimizer, criterion, finetune_data_loader, test_data_loader, 10000, args.train_loss_limit, 'FineTuning/')

if __name__ == '__main__':
    """
    Main function to train and fine-tune the MLP model.
    """
    parser = argparse.ArgumentParser(description='Train a 3-layer MLP for classification.')
    parser.add_argument('--width', type=int, default=2048, help='Width of the hidden layers.')
    parser.add_argument('--input_size', type=int, default=32, help='Input size.')
    parser.add_argument('--output_size', type=int, default=4, help='Number of classes for classification.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--train_loss_limit', type=float, default=1e-4, help='Loss threshold for early stopping.')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate.')
    parser.add_argument('--finetune_lr', type=float, default=1, help='Learning rate for fine-tuning.')
    parser.add_argument('--activation', choices=['ReLU', 'Linear'], default='Linear', help='Activation function.')
    parser.add_argument('--loss_type', choices=['MSE', 'CE'], default='CE', help='Loss type.')
    parser.add_argument('--num_samples', type=int, default=2048, help='Number of samples for initial training.')
    parser.add_argument('--fine_tuning_num_samples', type=int, default=20, help='Number of samples for initial training.')
    parser.add_argument('--parametrization', choices=['muP', 'NTK', 'SP', 'LP'], default='NTK', help='Type of parametrization.')
    
    parser.add_argument('--input_shift_cor', type=float, default=1, help='Input Cor.')
    parser.add_argument('--output_shift_cor', type=float, default=1, help='Output Cor.')
    parser.add_argument('--class_shift', action='store_true', default=False)

    parser.add_argument('--seed', type=int, default=123, help='seed')
    parser.add_argument('--wandb', action='store_true', default=False)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    main()
