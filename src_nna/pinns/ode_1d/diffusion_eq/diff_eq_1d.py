
import torch
import numpy as np
import time

from  src_nna.pinns.nn_pinns import MLP_PINN, EarlyStopper

from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(123)

"""
This script contains the implementation of the 1D diffusion equation
u + epsilon * u_xx =1
"""

def exact_solution(x, eps = 0.1):
    """
    exact solution of the 1D diffusion equation
    u(x) = 1 - cos(x/a) - sin(x/a) * tan(1/(2a))
    params:
    x: the input space
    a: float, the parameter of the problem
    """
    return  1-np.cos(x/eps)-np.sin(x/eps) * np.tan(1/(2*eps))

def loss_1d_diff_eq(model, x0_b, x1_b, x_phys, param):
    """
    loss function for the 1D diffusion equation
    loss = loss_boundary + loss_physics + loss_MSE
    params:
    model: torch.nn.Module, the neural network model
    x0_b: torch.tensor, the boundary at x=0
    x1_b: torch.tensor, the boundary at x=1
    x_phys: torch.tensor, the physical domain
    param: list, [epsilon] the parameters of the problem

    hyperparameters:
    lambda1, lambda2: float, the weights for the boundary loss
    """
    # using the following hyperparameters:
    lambda1, lambda2 = 1.5, 1.5 #1e-1, 1e-1
    
    # compute boundary loss
    u = model(x0_b)
    loss1 = (torch.squeeze(u) - 0)**2
    u = model(x1_b)
    loss2 = (torch.squeeze(u) - 0)**2

    # compute physics loss
    u = model(x_phys)
    dudx = torch.autograd.grad(u, x_phys, torch.ones_like(u), create_graph=True)[0]
    d2udx2 = torch.autograd.grad(dudx, x_phys, torch.ones_like(dudx), create_graph=True)[0]
    loss3 = torch.mean((param[0]**2 * d2udx2 + u - 1.0)**2)
    
    # backpropagate joint loss, take noptimizer step
    loss = loss3 + lambda1*loss1 + lambda2*loss2
        
    return loss

def train_1d_diff_eq(model, lear_rate, x0_b, x1_b, x_phys, x_test, param, num_iter, phys_errors,
                      val_errors, early_stopper=None, plotting=False):  

    """
    Train the PINN model to solve the 1D diffusion equation
    params:
    model: torch.nn.Module, the neural network model
    lear_rate: float, the learning rate
    x0_b: torch.tensor, the boundary at x=0
    x1_b: torch.tensor, the boundary at x=1
    x_phys: torch.tensor, the physical domain
    x_test: torch.tensor, the test domain
    param: list, [epsilon] the parameters of the problem
    num_iter: int, the number of iterations
    phys_errors: list, the physical errors
    val_errors: list, the validation errors
    plotting: bool, plot the result as training progresses
    """
    exact_sols= exact_solution(x_test[:,0], param[0])
    exact_sols_phys= exact_solution(x_phys.detach().numpy()[:,0], param[0])

    
    optimizer = torch.optim.Adam(model.parameters(),lr=lear_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.7)
    for i in tqdm(range(num_iter+1)):
        optimizer.zero_grad()
        
        # compute each term of the PINN loss function
        loss = loss_1d_diff_eq(model, x0_b, x1_b, x_phys, param)
        loss.backward()
        optimizer.step()

        u = model(x_phys)
        phys_err = torch.mean((u[:,0]-torch.tensor(exact_sols_phys))**2)
        phys_errors.append(phys_err.detach().numpy())


        u = model(x_test)
        test_err = torch.mean((u[:,0]-exact_sols)**2)
        val_errors.append(test_err.detach().numpy())
        scheduler.step()

        if (i > 100 and early_stopper):
            early_stopper.update(test_err, model)
            if early_stopper.early_stop:
                model = early_stopper.load_checkpoint(model)
                print('early stopping')
                break
        if phys_err < 1e-5:
            print(f"Converged at iteration {i}", "final error:", phys_err)
            break
    
        # plot the result as training progresses
        if (plotting and i % int(num_iter/2) == 1): 
            plot_pinn(x_test, x0_b, x1_b, u.detach().numpy(), i, param)

    if plotting:
        plot_pinn(x_test, x0_b, x1_b, u.detach().numpy(), i, param)





def plot_pinn(xs, x0_b, x1_b, u_pinn, step=0, param=[0.1]):
    """
    Plottting the PINN solution
    params:
    xs: torch.tensor, the input space
    x0_b: torch.tensor, the boundary at x=0
    x1_b: torch.tensor, the boundary at x=1
    u_pinn: torch.tensor, the PINN solution
    step: int, the current iteration
    param: list, [epsilon] the parameters of the problem
    """

    exact_sols = exact_solution(xs[:,0], param[0])
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    ax.scatter(x0_b.detach()[:,0], 
                torch.zeros_like(x0_b)[:,0], s=20, lw=0, color="red", alpha=0.95)
    ax.scatter(x1_b.detach()[:,0], 
                torch.zeros_like(x1_b)[:,0], s=20, lw=0, color="red", alpha=0.95)
    ax.plot(xs[:,0], u_pinn[:,0],  lw=2.5, label="PINN solution", color="tab:orange")
    ax.plot(xs[:,0], exact_sols,  "--",lw=1.5, label="Exact solution", color="black", alpha=1.)
    ax.set_xlabel("x")
    
    # ax.set_title(f"Training step: {step}")
    ax.legend()
    plt.show()
    # fig.savefig(f"./Figures/pinn/pinn_diff_eq_1d-trainstep{step}.pdf")


def plot_rror(phys_errrs, val_errrs):
    """
    Plot the loss function
    params:
    phys_errrs: list, the physics errors
    val_errrs: list, the validation errors
    """
    # plot the loss function
    fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
    ax.set_yscale("log")  
    ax.plot(np.array(phys_errrs), 'b-', label="Physics loss")
    ax.plot(np.array(val_errrs), '--r' , label="Test loss")
    ax.set_xlabel("Iterations")
    ax.legend()
    plt.show()
    # fig.savefig("./Figures/pinn/loss_diff_eq_1d.pdf")

def run_training(epsilon =0.1,   num_hidden=32, num_layers=3,
                  lear_rate= 1e-2,num_iteration = 7500, plotting=False):
    """
    Run the training of the PINN model to solve the 1D diffusion equation
    params:
    epsilon: float, the parameter of the problem
    num_hidden: int, the number of hidden units
    num_layers: int, the number of hidden layers
    lear_rate: float, the learning rate
    num_iteration: int, the number of iterations
    plotting: bool, plot the result as training progresses
    """
    # details of the network

    # num_hidden=32
    # num_layers=3
    # lear_rate= 1e-2
    # epsilon =0.1
    # num_iteration = 7500


    # define boundary points, for the boundary loss
    x0_boundary = torch.tensor(0.).view(-1,1).requires_grad_(True)
    x1_boundary = torch.tensor(1.).view(-1,1).requires_grad_(True)

    # define training points over the entire domain, for the physics loss
    x_physics = torch.linspace(0,1,100).view(-1,1).requires_grad_(True)
    x_test = torch.linspace(0,1,400).view(-1,1)

    # train the PINN
    start = time.time()
    phys_errs=[]
    test_errs=[]

    num_input=1
    num_output=1
    pinn = MLP_PINN(num_input, num_output, num_hidden, num_layers)
    early_stopper = EarlyStopper(verbose=False, path='checkpoint.pt', patience=500)
    loss = loss_1d_diff_eq(pinn, x0_boundary, x1_boundary, x_physics, [epsilon])
    train_1d_diff_eq(pinn, lear_rate,x0_boundary, x1_boundary, x_physics, x_test, [epsilon],
                      num_iteration, phys_errs, test_errs, early_stopper,  plotting=plotting)
    end = time.time()
    computation_time = {}
    computation_time["pinn"] = end - start
    print(f"computation time: {end-start:.4f}")

    plot_rror(phys_errs, test_errs)

    u_pinn = pinn(x_test).detach().numpy()
    plot_pinn(x_test, x0_boundary, x1_boundary, u_pinn, param=[epsilon])