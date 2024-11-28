
import torch
import numpy as np
import time

from scipy.stats import qmc

from  src_nna.pinns.nn_pinns import  EarlyStopping_val_train as EarlyStopping
from  src_nna.pinns.nn_pinns import MLP_PINN, EarlyStopper
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

torch.manual_seed(123)

"""
Here we solve the Heat equation in 2D
"""

def exact_solution(x, y, params):
    """
    Exact solution of the 2D heat equation
    T(x,y) = T1 + (Tm - T1) * sum( (1+(-1)**(n+1))/n * sin(n*pi*x/Lx) * sinh(n*pi*y/Ly) / sinh(n*pi*Ly/Lx) )
    params:
    x: the input space
    y: the input space
    params=[Lx, Ly, t1, tm]
        Lx: float, the length of the domain in the x direction
        Ly: float, the length of the domain in the y direction
        t1: float, the initial temperature
        tm: float, the maximum temperature
    """
    Lx, Ly, t1, tm = params
    nmax = 20

    tmp = 0
    for n in range(1, nmax):
        fac= (1+ (-1)**(n+1)) /n
        tmp += fac * np.sin(n*np.pi*x/Lx) * np.sinh(n*np.pi*y/Ly) / np.sinh(n*np.pi*Ly/Lx)
    tmp_prof = 2. * tmp / np.pi
    tmp_prof *= tm - t1
    tmp_prof += t1
    return tmp_prof


def loss_2d_heat_eq(model, x_d, y_d, t_d, x, y, params, verbose):
    """
    loss function for the 2D heat equation
    loss = loss_boundary + loss_physics + loss_MSE
    params:
    model: torch.nn.Module, the neural network model
    x_d: torch.tensor, the boundary for x-component
    y_d: torch.tensor, the boundary for y-component
    t_d: torch.tensor, the boundary values of temperature
    x: torch.tensor, the x-component physical domain inside the square
    y: torch.tensor, the y-component physical domain inside the square
    params: list, [Lx, Ly, t1, tm] the parameters of the problem
    """

    # boundary loss
    xy= torch.cat([x_d, y_d], dim=1)
    u_b = model(xy)
    loss_bound = torch.mean((t_d - u_b)**2)

    # Equation of motion loss and mse exact sol
    u_mse = model(torch.cat([x, y], dim=1))
    # loss exact solution
    ex_sol = exact_solution(x.detach().numpy(), y.detach().numpy(), params)
    loss_mse = torch.mean((torch.tensor(ex_sol) - u_mse)**2)

    # Equation of motion loss and mse exact sol
    u = model(torch.cat([x, y], dim=1))
    # Compute first-order gradients
    #torch.ones_like(u),or remove it and keep u.sum() to get the sum of u
    u_x = torch.autograd.grad(u, x, torch.ones_like(u),  create_graph=True)[0] 
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    
    # Compute second-order gradients
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    
    # Compute differential operator
    F = u_xx + u_yy

    loss_eom = torch.mean(F**2) # EOM loss
    
    # lam_bound = 1
    # lam_eom = np.power( 10, np.floor(np.log10(   loss_bound.detach().numpy()/ loss_eom.detach().numpy())))
    # lam_mse = np.power( 10, np.floor(np.log10(   loss_bound.detach().numpy()/ loss_mse.detach().numpy())))
    lam_eom, lam_bound, lam_mse = 1,0.5,1 #0, 1, 1
    loss = lam_eom* loss_eom+ lam_bound * loss_bound + lam_mse * loss_mse
    if verbose: print(loss_eom, loss_bound, loss_mse)
    # Return total loss
    return loss


def train_2d_heat_eq(model, lear_rate, x_d, y_d, t_d, x, y, x_t, y_t,
                      params, num_iter, phys_errors, test_errors, early_stopper=None, plotting=False, verbose=False):  
    
    """
    train the PINN model to solve the 2D heat equation
    params:
    model: torch.nn.Module, the neural network model
    lear_rate: float, the learning rate
    x_d: torch.tensor, the boundary for x-component
    y_d: torch.tensor, the boundary for y-component
    t_d: torch.tensor, the boundary values of temperature
    x: torch.tensor, the x-component physical domain inside the square
    y: torch.tensor, the y-component physical domain inside the square
    x_t: torch.tensor, the x-component test domain
    y_t: torch.tensor, the y-component test domain
    params: list, [Lx, Ly, t1, tm] the parameters of the problem
    num_iter: int, the number of iterations
    phys_errors: list, the physical errors
    test_errors: list, the validation errors
    early_stopper: EarlyStopper, the early stopping object
    plotting: bool, plot the result as training progresses
    """

    early_stopping = EarlyStopping(tolerance=3, min_delta=7)
    optimizer = torch.optim.Adam(model.parameters(),lr=lear_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                            factor=0.9, patience=5,
                                                              threshold=1e-8)
    lossess = []
    for i in tqdm(range(num_iter+1)):
        optimizer.zero_grad()
        
        # compute each term of the PINN loss function
        loss = loss_2d_heat_eq(model, x_d, y_d, t_d, x, y, params, verbose)
        lossess.append(loss.detach().numpy())
        loss.backward()
        optimizer.step()
        
        

        xy= torch.cat([x, y], dim=1)
        u = model(xy)
        exact_sols_phys = exact_solution(x.detach().numpy(), y.detach().numpy(), params)
        phys_err = torch.mean((u-torch.tensor(exact_sols_phys))**2)
        phys_errors.append(phys_err.detach().numpy())

        model.eval()
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            xy_t= torch.cat([x_t, y_t], dim=1)
            u_t = model(xy_t)
            exact_sols_phys = exact_solution(x_t.detach().numpy(), y_t.detach().numpy(), params)
            test_err = torch.mean((u_t-torch.tensor(exact_sols_phys))**2)
            test_errors.append(test_err.detach().numpy())
        
        scheduler.step(test_err)
        
        if verbose: print(optimizer.param_groups[0]['lr'])

        if (i > 100 and early_stopper):
                early_stopper.update(test_err, model)
                if early_stopper.early_stop:
                    model = early_stopper.load_checkpoint(model)
                    print('early stopping')
                    break
        if loss < 1e-9:
            print(f"Converged at iteration {i}", "final error:", loss)
            break
    
        # plot the result as training progresses
        if (plotting and i % int(num_iter/3) == 1): 
            plot_errors(phys_errors, test_errors, lossess)


def plot_errors(phys_errors, test_errors, lossess):

    """
    plot the loss functions in logscale
    params:
    phys_errors: list, the physical errors
    test_errors: list, the validation errors
    lossess: list, the total loss
    """

    plt.plot(phys_errors,'--', label="Physics Loss")
    plt.plot(test_errors,':', label="test Loss")
    plt.plot(lossess,'-.', label="all Loss")
    plt.yscale('log')
    plt.legend()
    plt.show()

def plot_exactsol(X,Y, params):
    """
    plot the exact solution of the 2D heat equation
    params:
    X: np.array, the x-component input space
    Y: np.array, the y-component input space
    params: list, [Lx, Ly, t1, tm] the parameters of the problem
    """

    exact_sol = exact_solution(X, Y, params)
    plt.contourf(X, Y, exact_sol, 20, cmap='coolwarm')
    plt.colorbar()
    plt.show()


def run_training( Lx=1, Ly=1, t1=0.1, tm=1.0, n_bc = 4, n_data_per_bc = 1000,
                    Nc = 600,lear_rate=  5e-2, num_iteration = 1600,
                    n = 1000, num_hidden=16, num_layers=4,  plotting = False, verbose= False):
    """
    Run the training of the PINN model to solve the 2D heat equation
    params:
    Lx: float, the length of the domain in the x direction
    Ly: float, the length of the domain in the y direction
    t1: float, the initial temperature
    tm: float, the maximum temperature
    n_bc: int, the number of boundary conditions
    n_data_per_bc: int, the number of data points per boundary condition
    Nc: int, the number of collocation points
    lear_rate: float, the learning rate
    num_iteration: int, the number of iterations
    n: int, the number of points to plot the result
    num_hidden: int, the number of hidden units
    num_layers: int, the number of layers
    plotting: bool, plot the result
    verbose: bool, print the loss function
    """

    ### data generation
    # plotting = False
    # n_bc = 4
    # n_data_per_bc = 1000
    # Lx=1
    # Ly=1
    # t1=0.1
    # tm=1.0
    params=[Lx, Ly, t1, tm]
    #
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_bc, n_data_per_bc, 3])

    for i, j in zip(range(n_bc), [0, Lx, 0, Ly]):
        points = (engine.random(n=n_data_per_bc)[:, 0] ) 
        if i < 2:
            data[i, :, 0] = j #x component, x boundary
            data[i, :, 1] = points #y component, x boundary
        else:
            data[i, :, 0] = points #x component, yboundary
            data[i, :, 1] = j #y component, y boundary

    # boundary condition Values
    data[0, :, 2] = t1
    data[1, :, 2] = t1
    data[2, :, 2] = t1
    data[3, :, 2] = tm * np.sin(np.pi * data[3, :, 0])

    data = data.reshape(n_data_per_bc * n_bc, 3)
    #
    x_d, y_d, t_d = map(lambda x: np.expand_dims(x, axis=1), 
                        [data[:, 0], data[:, 1], data[:, 2]])

#
    # sampling from points inside the square
    engine = qmc.LatinHypercube(d=2)
    colloc = engine.random(n=Nc)*Lx
    colloc_test = engine.random(n=2*Nc)*Ly
    #
    x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), 
                [colloc[:, 0], colloc[:, 1]])
    x_t, y_t = map(lambda x: np.expand_dims(x, axis=1), 
                [colloc_test[:, 0], colloc_test[:, 1]])

    #

    if plotting:
        plt.figure(figsize=(5, 5))
        plt.title("Boundary Data points and Collocation points")
        plt.scatter(data[:, 0], data[:, 1], marker="x", c="k", label="boundary points")
        plt.scatter(colloc[:, 0], colloc[:, 1], s=1, marker="o", c="r", label="inside points")
        plt.scatter(colloc_test[:, 0], colloc_test[:, 1], s=1, marker="*", c="blue", label="test inside points")
        plt.legend()
        plt.show()

    x_c, y_c, x_d, y_d, t_d =map(lambda x: torch.tensor(x,dtype=torch.float32).requires_grad_(True),
                                [x_c, y_c, x_d, y_d, t_d])
    
    x_t, y_t =map(lambda x: torch.tensor(x,dtype=torch.float32),
                                [x_t, y_t])
    
    start = time.time()
    #for deeper network, increase the number of samples inside the
    num_input, num_output = (2, 1)
    # num_input, num_output, num_hidden, num_layers = (2, 1, 16, 4) #(2, 1, 32, 4)# (2, 1, 36, 3)#(2, 1, 64, 2) #(2, 1, 10, 4) # 
    pinn = MLP_PINN(num_input, num_output, num_hidden, num_layers)
    early_stopper = EarlyStopper(verbose=False, path='checkpoint.pt', patience=150)

    print("start training: ")
    phys_errs=[]
    test_errs=[]
    train_2d_heat_eq(pinn, lear_rate, x_d, y_d, t_d, x_c, y_c, x_t, y_t, params,
                    num_iteration, phys_errs, test_errs, early_stopper, plotting=plotting, verbose=verbose)

    end = time.time()
    computation_time = {}
    computation_time["pinn"] = end - start
    print(f"computation time: {end-start:.4f}")

    fig, axs = plt.subplots(1, 1, figsize=(5, 2.6))
    axs.semilogy(phys_errs)
    axs.set_xlabel("iteration")
    axs.set_ylabel("Loss")
    plt.show()
    # plt.savefig("./Figures/pinn/loss_heat_eq_2d.pdf")


    X = np.linspace(0, Lx, n)
    Y = np.linspace(0, Ly, n)
    X0, Y0 = np.meshgrid(X, Y)
    X = X0.reshape([n*n, 1])
    Y = Y0.reshape([n*n, 1])
    X_T = torch.tensor(X, dtype=torch.float32)
    Y_T = torch.tensor(Y, dtype=torch.float32)
    xy= torch.cat([X_T, Y_T], dim=1)

    S = pinn(xy)
    S = S.detach().numpy().reshape(n, n)
    #
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.))
    im2 = axs[0].pcolormesh(X0, Y0, exact_solution(X0, Y0, params), vmax=tm, vmin=t1, cmap="coolwarm")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("Exact solution")
    axs[0].set_aspect("equal")

    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im2, cax=cax1)
    cbar1.set_label('exact solution')

    im1 = axs[1].pcolormesh(X0, Y0, S, vmax=tm, vmin=t1, cmap="coolwarm")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title("PINNs")
    axs[1].set_aspect("equal")

    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    cbar1.set_label('PINNs')

    vmax= np.max(np.abs(exact_solution(X0, Y0, params)-S))
    im3 = axs[2].pcolormesh(X0, Y0, exact_solution(X0, Y0, params)-S, vmin=-vmax, vmax=vmax, cmap="seismic")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title("Difference")
    axs[2].set_aspect("equal")

    divider1 = make_axes_locatable(axs[2])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im3, cax=cax1)
    cbar1.set_label('Difference')

    plt.show()