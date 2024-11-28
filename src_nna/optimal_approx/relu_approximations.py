import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ReLUNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
        # Kaiming initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_network(func, x_range, hidden_neurons, epochs=10000, patience=100):
    x = torch.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
    y = func(x)
    
    model = ReLUNetwork(1, hidden_neurons)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    no_improve = 0
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss < best_loss:
            best_loss = loss
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    return model, losses

def compute_errors(func, model, x_range):
    x = torch.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
    y_true = func(x)
    y_pred = model(x)
    
    l1_error = torch.max(torch.abs(y_true - y_pred)).item()
    mse = torch.mean((y_true - y_pred)**2).item()
    
    return l1_error, np.sqrt(mse)

def plot_function_and_approximation(func, model, x_range, title, ax):
    x = torch.linspace(x_range[0], x_range[1], 1000).reshape(-1, 1)
    y_true = func(x)
    y_pred = model(x)
    
    ax.plot(x, y_true, label='True function')
    ax.plot(x, y_pred.detach(), label='ReLU approximation')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True)

functions = {
    'exp': (torch.exp, (0, 1)),
    'square': (lambda x: x**2, (-1, 1)),
    'cube': (lambda x: x**3, (0, 1))
}

neuron_mapping = {
    'exp': {2: 5, 3: 8, 5: 10, 10: 36},
    'square': {2: 3, 3: 4, 5: 11, 10: 24},
    'cube': {2: 3, 3: 5, 5: 17, 10: 36}
}

results = {}

fig_approx, axes_approx = plt.subplots(len(functions), 4, figsize=(20, 5*len(functions)))
fig_approx.suptitle('Function Approximations (Kaiming Initialization)', fontsize=16)

fig_rmse, axes_rmse = plt.subplots(len(functions), 1, figsize=(10, 5*len(functions)))
fig_rmse.suptitle('RMSE Reduction (Kaiming Initialization)', fontsize=16)

for i, (func_name, (func, x_range)) in enumerate(functions.items()):
    results[func_name] = {}
    for j, n in enumerate([2, 3, 5, 10]):
        hidden_neurons = neuron_mapping[func_name][n]
        
        model, losses = train_network(func, x_range, hidden_neurons)
        l1_error, rmse = compute_errors(func, model, x_range)
        
        results[func_name][n] = {
            'hidden_neurons': hidden_neurons,
            'L1_error': l1_error,
            'RMSE': rmse,
            'losses': losses
        }
        
        plot_function_and_approximation(func, model, x_range, 
                                        f'{func_name} with {n} segments', axes_approx[i, j])
        
        axes_rmse[i].plot(range(len(losses)), losses, label=f'n={n}')
        axes_rmse[i].set_title(f'{func_name} RMSE Reduction')
        axes_rmse[i].set_xlabel('Epochs')
        axes_rmse[i].set_ylabel('RMSE')
        axes_rmse[i].set_yscale('log')
        axes_rmse[i].grid(True)
        axes_rmse[i].legend()

for func_name, func_results in results.items():
    print(f"\nResults for {func_name} function:")
    for n, data in func_results.items():
        print(f"  n (number of segments): {n}")
        print(f"    Hidden neurons: {data['hidden_neurons']}")
        print(f"    L1 error: {data['L1_error']:.6f}")
        print(f"    RMSE: {data['RMSE']:.6f}")

plt.tight_layout()
plt.show()

fig_error, axes_error = plt.subplots(len(functions), 1, figsize=(10, 5*len(functions)))
fig_error.suptitle('Errors vs Number of Segments (Kaiming Initialization)', fontsize=16)

for i, (func_name, func_results) in enumerate(results.items()):
    segments = list(func_results.keys())
    l1_errors = [func_results[s]['L1_error'] for s in segments]
    rmses = [func_results[s]['RMSE'] for s in segments]
    
    axes_error[i].plot(segments, l1_errors, 'o-', label='L1 Error')
    axes_error[i].plot(segments, rmses, 's-', label='RMSE')
    axes_error[i].set_title(f'{func_name} Errors vs Number of Segments')
    axes_error[i].set_xlabel('Number of segments (n)')
    axes_error[i].set_ylabel('Error')
    axes_error[i].legend()
    axes_error[i].set_yscale('log')
    axes_error[i].grid(True)

plt.tight_layout()
plt.show()