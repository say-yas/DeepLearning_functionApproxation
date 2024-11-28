import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ReLUNetwork(nn.Module):
    "Defines a standard fully-connected network in PyTorch"
    
    def __init__(self, num_input, num_output, num_hidden, num_layers):
        super().__init__()
        activation = nn.Tanh
        self.fcs = nn.Sequential(*[
                        nn.Linear(num_input, num_hidden),
                        nn.ReLU()])
        self.fch = nn.Sequential(*[
                        nn.Sequential(*[
                            nn.Linear(num_hidden, num_hidden),
                            nn.ReLU()]) for _ in range(num_layers-1)])
        self.fce = nn.Linear(num_hidden, num_output)

    def forward(self, x):
        x = self.fcs(x)
        x = self.fch(x)
        x = self.fce(x)
        return x
    
class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if (val_loss) > self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
    
# class ReLUNetwork(nn.Module):
#     def __init__(self, num_neurons):
#         super(ReLUNetwork, self).__init__()
#         self.layer1 = nn.Linear(1, num_neurons)
#         self.relu = nn.ReLU()
#         self.layer2 = nn.Linear(num_neurons, 1)

#     def forward(self, x):
#         x = self.relu(self.layer1(x))
#         return self.layer2(x)

def convex_function(x):
    return x**2  # Example convex function

def train_network(model, x_train, y_train, errs, epochs=1000, learning_rate=1e-2):
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(tolerance=3, min_delta=6)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        errs.append(loss.item())
        scheduler.step(loss.item())

        early_stopping(loss)
        if (early_stopping.early_stop and epoch > 100):       
            break

        if loss.item() < 1e-5:
            print(f"Converged at iteration {epoch}", "final error:", loss.item())
            break

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")


try:
    # Generate training data
    x_train = torch.linspace(-1, 1, 100).reshape(-1, 1)
    y_train = torch.tensor(convex_function(x_train.numpy()), dtype=torch.float32)

    # Create and train the model
    model = ReLUNetwork(1, 1, 16, 2)
    errors = []
    train_network(model, x_train, y_train, errors, epochs=3000)

    plt.figure(figsize=(7,3.5))
    plt.yscale("log")  
    plt.grid()
    plt.plot(np.array(errors), 'b-', label="Loss")
    plt.show()

    # Generate test data for plotting
    x_test = torch.linspace(-1.1, 1.1, 200).reshape(-1, 1)
    y_test = convex_function(x_test.numpy())
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test).numpy()

    # Plotting
    plt.figure(figsize=(7, 4))
    plt.xlim(-1.1, 1.1)
    plt.plot(x_test.numpy(), y_test, label='True Function')
    plt.plot(x_test.numpy(), y_pred, label='ReLU Network Approximation')
    plt.scatter(x_train.numpy(), y_train.numpy(), color='red', s=10, label='Training Data')
    plt.legend()
    plt.title('ReLU Network Approximation of a Convex Function (PyTorch)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

except:
    raise ValueError("Error in training the RELU network pytorch model")