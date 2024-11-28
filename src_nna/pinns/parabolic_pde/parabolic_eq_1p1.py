import torch
import time
import numpy as np
from scipy.stats import qmc

from  src_nna.pinns.nn_pinns import MLP_PINN, EarlyStopper
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(123)

"""
This file contains the class Parabolic_1p1_pinns to solve the 1+1 parabolic equation
"""

def cos(x):
    """
    cos function
    """
    return np.cos(x)

def sin(x):
    """
    sin function
    """
    return np.sin(x)

def x_sqrt(x):
    """"
    sqrt function
    """
    return np.sqrt(x)

def x_pow2(x):
    """
    x^2 function
    """
    return np.power(x,2)

def func_in( x,  n, func):
    """
    auxiliary function to compute the coefficients of the Fourier series
    """
    return func(x)*np.sin(n*np.pi*x)


def exact_solution(x, t, func, alpha=1.0, L=1.0):
    """
    exact solution of the parabolic 1+1 space-time dimensional PDE
    Args:
    x: float, the space domain
    t: float, the time domain
    func: function, the boundary condition function
    alpha: float, the diffusion coefficient
    L: float, the length of the space domain
    """
    def compute_an(func, L, n):
        from  scipy import integrate 
        res = integrate.quad( func_in, 0, L, args=(n, func))
        return 2.0 * res[0]

    utx = 0
    for n in np.arange(1, 10):
        an = compute_an(func, L, n) 
        utx += an * np.exp(- t * (n*np.pi * alpha)**2 ) * np.sin(n*np.pi*x)
    return  utx


class Parabolic_1p1_pinns:
    """
    Class to train the PINN model to solve the 1+1 parabolic equation
    Args:
    alpha: float, the diffusion coefficient
    Lx: float, the length of the space domain
    Lt: float, the length of the time domain
    func: function, the boundary condition function
    x_d: torch tensor, the x boundary condition
    t_d: torch tensor, the t boundary condition
    bound_d: torch tensor, the boundary condition values
    x: torch tensor, the collocation points in the space domain
    t: torch tensor, the collocation points in the time domain
    x_test: torch tensor, the test points in the space domain
    t_test: torch tensor, the test points in the time domain
    lear_rate: float, the learning rate
    lam_eom: float, the weight of the equation of motion loss
    lam_bound: float, the weight of the boundary loss
    lam_mse: float, the weight of the MSE loss
    verbose: bool, print the loss values
    """
    def __init__(self, alpha, Lx, Lt, func,  x_d, t_d, bound_d, x, t,
                  x_test, t_test, lear_rate=0.001, lam_eom=1.0, lam_bound=1.0,
                    lam_mse=1.0 , verbose=False):
        self.alpha = alpha
        self. Lx = Lx
        self.Lt =Lt

        self.func = func

        self.lear_rate = lear_rate
        self.lam_eom, self.lam_bound, self.lam_mse = lam_eom, lam_bound, lam_mse 

        self.x_d, self.t_d, self.bound_d, self.x, self.t =  x_d, t_d, bound_d, x, t

        self.x_test, self.t_test = x_test, t_test

        self.verbose = verbose



    def loss_parabolic_1p1(self, model):
        """
        loss function for the 2D heat equation
        loss = loss_boundary + loss_physics + loss_MSE
        params:
        model: torch model, the model to train
        """

        # boundary loss
        xt= torch.cat([self.x_d, self.t_d], dim=1)
        u_b = model(xt)
        loss_bound = torch.mean((self.bound_d - u_b)**2)

        # Equation of motion loss and mse exact sol
        u_mse = model(torch.cat([self.x, self.t], dim=1))
        # loss exact solution
        ex_sol = exact_solution(self.x.detach().numpy(), self.t.detach().numpy(), self.func, self.alpha, self.Lx)
        loss_mse = torch.mean((torch.tensor(ex_sol) - u_mse)**2)

        # Equation of motion loss and mse exact sol
        u = model(torch.cat([self.x, self.t], dim=1))
        # Compute first-order gradients
        #torch.ones_like(u),or remove it and keep u.sum() to get the sum of u
        u_x = torch.autograd.grad(u, self.x, torch.ones_like(u),  create_graph=True)[0] 
        u_t = torch.autograd.grad(u, self.t, torch.ones_like(u), create_graph=True)[0]
        
        # Compute second-order gradients
        u_xx = torch.autograd.grad(u_x, self.x, torch.ones_like(u_x), create_graph=True)[0]
        
        # Compute differential operator
        F =  u_t - self.alpha**2 * u_xx 

        loss_eom = torch.mean(F**2) # EOM loss

        loss = self.lam_eom* loss_eom+ self.lam_bound * loss_bound + self.lam_mse * loss_mse
        if self.verbose:
            print(loss_eom, loss_bound, loss_mse)
        # Return total loss
        return loss
    
    def train_parabolic_1p1(self, model, num_iter, phys_errors, test_errors,
                             early_stopper=None, plotting=False):  
        
        """
        train the PINN model to solve the 1_1 parabolic equation
        params:
        model: torch model, the model to train
        num_iter: int, number of iterations
        phys_errors: list, the physical errors
        test_errors: list, the validation errors
        early_stopper: EarlyStopper, the early stopping object
        plotting: bool, plot the results
        """

        optimizer = torch.optim.Adam(model.parameters(),lr= self.lear_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                factor=0.95, patience=40,
                                                                threshold=1e-8)
        lossess = []
        for i in tqdm(range(num_iter+1)):
            optimizer.zero_grad()
            model.train()
            
            # compute each term of the PINN loss function
            loss = self.loss_parabolic_1p1(model)
            lossess.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            
            xt= torch.cat([self.x, self.t], dim=1)
            u = model(xt)
            exact_sols_phys = exact_solution(self.x.detach().numpy(), self.t.detach().numpy(),
                                              self.func, self.alpha, self.Lx)
            phys_err = torch.mean((u-torch.tensor(exact_sols_phys))**2)
            phys_errors.append(phys_err.detach().numpy())

            model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                xt_test= torch.cat([self.x_test, self.t_test], dim=1)
                u_t = model(xt_test)
                exact_sols_phys = exact_solution(self.x_test.detach().numpy(), self.t_test.detach().numpy(),
                                              self.func, self.alpha, self.Lx)
                test_err = torch.mean((u_t-torch.tensor(exact_sols_phys))**2)
                test_errors.append(test_err.detach().numpy())
            
            scheduler.step(phys_err)
            
            if self.verbose: print(optimizer.param_groups[0]['lr'])


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
                self.plot_errors(phys_errors, test_errors, lossess)

    def plot_errors(self, phys_errors, test_errors, lossess):

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


def run_traning(n_bc = 3, Lx=1.0, Lt=0.7, alpha = 0.5, Nc = 700 , n = 700, num_hidden=32, num_layers=8,
                 lear_rate= 5e-3, num_iteration = 1600, function_bound = cos, n_data_per_bc = 2000, plotting = False):

    """
    Run the training of the PINN model to solve the 1+1 parabolic equation
    params:
    n_bc: int, number of boundaries
    Lx: float, the length of the space domain
    Lt: float, the length of the time domain
    alpha: float, the diffusion coefficient
    Nc: int, number of collocation points
    n: int, number of points to evaluate the solution
    num_hidden: int, number of hidden units
    num_layers: int, number of hidden layers
    lear_rate: float, the learning rate
    num_iteration: int, number of iterations
    function_bound: function, the boundary condition function
    n_data_per_bc: int, number of data points per boundary condition
    plotting: bool, plot the results
    """
    params=[Lx, Lt, alpha]
    #
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_bc, n_data_per_bc, 3])

    for i, j in zip(range(n_bc), [0, Lx, 0]):
        points = (engine.random(n=n_data_per_bc)[:, 0] ) 
        if i < 2:
            data[i, :, 0] = j #x component, x boundary u(0, t)
            data[i, :, 1] = points*Lt #t component, x boundary u(Lx,t)
        else:
            data[i, :, 0] = points*Lx #x component, t boundary u(x,0)
            data[i, :, 1] = j #t component, t boundary u(x,Lt)

    # boundary condition Values
    data[0, :, 2] = 0
    data[1, :, 2] = 0
    data[2, :, 2] = function_bound(data[2, :, 0])

    data = data.reshape(n_data_per_bc * n_bc, 3)
    #
    x_d, t_d, bound_d = map(lambda x: np.expand_dims(x, axis=1), 
                        [data[:, 0], data[:, 1], data[:, 2]])

#
    # sampling from points inside the square
    engine = qmc.LatinHypercube(d=2)
    colloc = engine.random(n=Nc)
    colloc_test = engine.random(n=2*Nc)
    #
    x_c, t_c = map(lambda x: np.expand_dims(x, axis=1), 
                [colloc[:, 0]*Lx, colloc[:, 1]*Lt])
    x_t, t_t = map(lambda x: np.expand_dims(x, axis=1), 
                [colloc_test[:, 0]*Lx, colloc_test[:, 1] *Lt])

    #

    if plotting:
        plt.figure(figsize=(5, 5))
        plt.title("Boundary Data points and Collocation points")
        plt.scatter(data[:, 0], data[:, 1], marker="x", c=data[:, 2], label="boundary points")
        plt.scatter(colloc[:, 0]*Lx, colloc[:, 1]*Lt, s=10, marker="o", c="r", label="inside points")
        plt.scatter(colloc_test[:, 0]*Lx, colloc_test[:, 1]*Lt, s=1, marker="*", c="blue", label="test inside points")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()
        plt.show()

    x_c, t_c, x_d, t_d, bound_d =map(lambda x: torch.tensor(x,dtype=torch.float32).requires_grad_(True),
                                [x_c, t_c, x_d, t_d, bound_d])
    
    x_t, t_t =map(lambda x: torch.tensor(x,dtype=torch.float32),
                                [x_t, t_t])
    
    start = time.time()
    #for deeper network, increase the number of samples inside the
    num_input, num_output = (2, 1)
    # num_hidden, num_layers = #( 2**7, 6) #( 32, 8) #(24, 4) #(10, 6) #(64, 6) #(64, 4)  
    pinn = MLP_PINN(num_input, num_output, num_hidden, num_layers)

    early_stopper = EarlyStopper(verbose=False, path='checkpoint.pt', patience=100)

    print("start training: ")
    phys_errs=[]
    test_errs=[]
    parab_pinns = Parabolic_1p1_pinns(alpha, Lx, Lt, function_bound,  x_d, t_d, bound_d, x_c, t_c,
                  x_t, t_t, lear_rate, lam_eom=1.0, lam_bound=0.5, lam_mse=1.0)
    parab_pinns.train_parabolic_1p1(pinn, num_iteration, phys_errs, test_errs,
                                     early_stopper= early_stopper, plotting=plotting)

    end = time.time()
    computation_time = {}
    computation_time["pinn"] = end - start
    print(f"computation time: {end-start:.4f}")

    fig, axs = plt.subplots(1, 1, figsize=(5, 2.6))
    axs.semilogy(phys_errs, label="Physics Loss")
    axs.set_xlabel("iterations")
    axs.set_ylabel("Error")
    plt.show()

    
    
    X = np.linspace(0, Lx, n)
    T = np.linspace(0, Lt, n)
    X0, T0 = np.meshgrid(X, T)
    X = X0.reshape([n*n, 1])
    T = T0.reshape([n*n, 1])
    X_T = torch.tensor(X, dtype=torch.float32)
    T_t = torch.tensor(T, dtype=torch.float32)
    xt= torch.cat([X_T, T_t], dim=1)
    S = pinn(xt)
    S = S.detach().numpy().reshape(n, n)

    #
    fig, axs = plt.subplots(1, 3, figsize=(13, 2.6))
    vmax= np.max(np.abs(exact_solution(X0, T0, function_bound, alpha=alpha, L=Lx)))
    im2 = axs[0].pcolormesh(X0, T0, exact_solution(X0, T0, function_bound, alpha=alpha, L=Lx),
                             vmax=vmax,   cmap="coolwarm")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    axs[0].set_title("Exact solution")
    axs[0].set_aspect("equal")

    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im2, cax=cax1)
    # cbar1.set_label('exact solution')

    im1 = axs[1].pcolormesh(X0, T0, S, vmax=vmax, cmap="coolwarm")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")
    axs[1].set_title("PINNs")
    axs[1].set_aspect("equal")

    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    # cbar1.set_label('PINNs')

    vmax= np.max(np.abs(exact_solution(X0, T0, function_bound, alpha=alpha, L=Lx)-S))
    im3 = axs[2].pcolormesh(X0, T0, exact_solution(X0, T0, function_bound, alpha=alpha, L=Lx)-S,
                             vmin=-vmax, vmax=vmax, cmap="seismic")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("t")
    axs[2].set_title("Difference")
    axs[2].set_aspect("equal")

    divider1 = make_axes_locatable(axs[2])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im3, cax=cax1)
    # cbar1.set_label('Difference')

    plt.show()



