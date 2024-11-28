import torch
import time
import numpy as np
from scipy.stats import qmc

from  src_nna.pinns.nn_pinns import MLP_PINN, EarlyStopper
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(123)

def cos(x):
    """
    cosine function
    """
    return np.cos(x)

def sin(x):
    """
    sine function
    """
    return np.sin(x)

def x_sqrt(x):
    """
    x_sqrt function
    """
    return np.sqrt(x)

def x_pow2(x):
    """x^2 function"""
    return np.power(x,2)

def x_lin(x):
    """x function"""
    return x

def cnst(x, constant=1):
    """constant function"""
    return constant

def func_in( x,  n, func):
    """auxiliary function for computing the fourier coefficients"""
    return func(x)*np.sin(n*np.pi*x)


def exact_solution(x, t, func, func_dudt, alpha=1.0, L=1.0):
    """
    exact solution of the Hyperbolic 1+1 space-time dimensional PDE
    Args:
    x: np.array, the spatial points
    t: np.array, the time points
    func: function, the function to be solved
    func_dudt: function, the function of the derivative of the function
    alpha: float, the constant
    L: float, the length of the domain
    """
    def compute_an_bn(func, func_dudt, L, n):
        from  scipy import integrate 
        an = integrate.quad( func_in, 0, L, args=(n, func))
        bn = integrate.quad( func_in, 0, L, args=(n, func_dudt))
        return 2.0 * an[0], 2*bn[0]/(n*np.pi)

    utx = 0
    for n in np.arange(1, 10):
        an, bn = compute_an_bn(func, func_dudt, L, n) 
        tmp_bn = bn * np.cos(n*np.pi * alpha * t) * np.sin(n*np.pi*x)
        tmp_an = an * np.sin(n*np.pi * alpha * t) * np.sin(n*np.pi*x)
        utx +=  (tmp_bn +tmp_an)
    return  utx


class Hyperbolic_1p1_pinns:
    """
    Class for solving the 1+1 Hyperbolic PDE using PINNs
    Params:
    alpha: float, the constant
    Lx: float, the length of the domain in the x direction
    Lt: float, the length of the domain in the t direction
    func: function, the function to be solved
    func_dudt: function, the function of the derivative of the function
    x_d: torch.tensor, the x boundary points
    t_d: torch.tensor, the t boundary points
    bound_d: torch.tensor, the boundary values
    x_dudt_d: torch.tensor, the x boundary points for the derivative
    t_dudt_d: torch.tensor, the t boundary points for the derivative
    bound_dudt_d: torch.tensor, the boundary values for the derivative
    x: torch.tensor, the x points inside the domain
    t: torch.tensor, the t points inside the domain
    x_test: torch.tensor, the x points for testing
    t_test: torch.tensor, the t points for testing
    lear_rate: float, the learning rate
    lam_eom: float, the weight for the EOM loss
    lam_bound: float, the weight for the boundary loss
    lam_mse: float, the weight for the MSE loss
    verbose: bool, the verbosity
    """
    def __init__(self, alpha, Lx, Lt, func, func_dudt,  x_d, t_d, bound_d, x_dudt_d, t_dudt_d, bound_dudt_d, x, t,
                  x_test, t_test, lear_rate=0.001, lam_eom=1.0, lam_bound=1.0, lam_mse=1.0 , verbose=False):
        self.alpha = alpha
        self. Lx = Lx
        self.Lt =Lt

        self.func = func
        self.func_dudt = func_dudt

        self.lear_rate = lear_rate
        self.lam_eom, self.lam_bound, self.lam_mse = lam_eom, lam_bound, lam_mse 

        self.x_d, self.t_d, self.bound_d, self.x, self.t = x_d, t_d, bound_d, x, t
        self.x_dudt_d, self.t_dudt_d, self.bound_dudt_d = x_dudt_d, t_dudt_d, bound_dudt_d

        self.x_test, self.t_test = x_test, t_test

        self.verbose = verbose



    def loss_Hyperbolic_1p1(self, model):
        """
        loss function for the 2D heat equation
        loss = loss_boundary + loss_physics + loss_MSE
        params:
        model: torch.nn.Module, the model to be trained
        """

        # boundary loss
        xt= torch.cat([self.x_d, self.t_d], dim=1)
        u_b = model(xt)
        loss_bound = torch.mean((self.bound_d - u_b)**2)


        u_ut = model(torch.cat([self.x_dudt_d, self.t_dudt_d], dim=1))
        u_t_d = torch.autograd.grad(u_ut, self.t_dudt_d, torch.ones_like(u_ut), create_graph=True)[0]
        loss_bound  += torch.mean((self.bound_dudt_d - u_t_d)**2)

        # Equation of motion loss and mse exact sol
        u_mse = model(torch.cat([self.x, self.t], dim=1))
        # loss exact solution
        ex_sol = exact_solution(self.x.detach().numpy(), self.t.detach().numpy(), self.func, self.func_dudt, self.alpha, self.Lx)
        loss_mse = torch.mean((torch.tensor(ex_sol) - u_mse)**2)

        # Equation of motion loss and mse exact sol
        u = model(torch.cat([self.x, self.t], dim=1))
        # Compute first-order gradients
        #torch.ones_like(u),or remove it and keep u.sum() to get the sum of u
        u_x = torch.autograd.grad(u, self.x, torch.ones_like(u),  create_graph=True)[0] 
        u_t = torch.autograd.grad(u, self.t, torch.ones_like(u), create_graph=True)[0]
        
        # Compute second-order gradients
        u_xx = torch.autograd.grad(u_x, self.x, torch.ones_like(u_x), create_graph=True)[0]
        u_tt = torch.autograd.grad(u_t, self.t, torch.ones_like(u_t), create_graph=True)[0]
        
        # Compute differential operator
        F =  u_tt - self.alpha**2 * u_xx 

        loss_eom = torch.mean(F**2) # EOM loss

        loss = self.lam_eom* loss_eom+ self.lam_bound * loss_bound + self.lam_mse * loss_mse
        if self.verbose:
            print(loss_eom, loss_bound, loss_mse)
        # Return total loss
        return loss
    
    def train_Hyperbolic_1p1(self, model, num_iter, phys_errors, test_errors, early_stopper=None,
                              plotting=False):  
        
        """
        train the PINN model to solve the 1_1 Hyperbolic equation
        params:
        model: torch.nn.Module, the model to be trained
        num_iter: int, the number of iterations
        phys_errors: list, the physical errors
        test_errors: list, the validation errors
        early_stopper: EarlyStopper, the early stopping class
        plotting: bool, the verbosity
        """

        optimizer = torch.optim.Adam(model.parameters(),lr= self.lear_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                factor=0.9, patience=10,
                                                                threshold=1e-8)
        lossess = []
        for i in tqdm(range(num_iter+1)):
            optimizer.zero_grad()
            model.train()
            
            # compute each term of the PINN loss function
            loss = self.loss_Hyperbolic_1p1(model)
            lossess.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            
            xt= torch.cat([self.x, self.t], dim=1)
            u = model(xt)
            exact_sols_phys = exact_solution(self.x.detach().numpy(), self.t.detach().numpy(),
                                              self.func, self.func_dudt, self.alpha, self.Lx)
            phys_err = torch.mean((u-torch.tensor(exact_sols_phys))**2)
            phys_errors.append(phys_err.detach().numpy())

            model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                xt_test= torch.cat([self.x_test, self.t_test], dim=1)
                u_t = model(xt_test)
                exact_sols_phys = exact_solution(self.x_test.detach().numpy(), self.t_test.detach().numpy(),
                                              self.func, self.func_dudt, self.alpha, self.Lx)
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


def run_traning(n_bc = 3, Lx=1.0, Lt=0.7, alpha = 0.5, Nc = 700 , n = 700, num_hidden=12, num_layers=8,
                 lear_rate= 5e-3, num_iteration = 1600, function_bound = x_pow2,
                   function_dudt_bound = x_lin, n_data_per_bc = 2000, plotting = False, verbose= False,
                     lam_eom=1, lam_bound=0.5, lam_mse=1):
    """
    Run the training of the PINN model for the Hyperbolic 1+1 PDE
    params:
    n_bc: int, the number of boundary conditions
    Lx: float, the length of the domain in the x direction
    Lt: float, the length of the domain in the t direction
    alpha: float, the constant
    Nc: int, the number of points inside the domain
    n: int, the number of points for the meshgrid
    num_hidden: int, the number of hidden units
    num_layers: int, the number of layers
    lear_rate: float, the learning rate
    num_iteration: int, the number of iterations
    function_bound: function, the function to be solved
    function_dudt_bound: function, the function of the derivative of the function
    n_data_per_bc: int, the number of data points per boundary condition
    plotting: bool, the verbosity
    verbose: bool, the verbosity
    lam_eom: float, the weight for the EOM loss
    lam_bound: float, the weight for the boundary loss
    lam_mse: float, the weight for the MSE loss
    """

    params=[Lx, Lt, alpha]
    #
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_bc, n_data_per_bc, 3])
    data_bound_dudt = np.zeros([1, n_data_per_bc, 3])

    for i, j in zip(range(n_bc), [0, Lx, 0]):
        points = (engine.random(n=n_data_per_bc)[:, 0] ) 
        if i < 2:
            data[i, :, 0] = j #x component, x boundary x=0,Lx
            data[i, :, 1] = points*Lt #t component, x boundary x=0,Lx
        else:
            data[i, :, 0] = points*Lx #x component, t boundary t=0
            data[i, :, 1] = j #t component, t boundary t=0

        data_bound_dudt[0, :, 0] = data[2, :, 0]
        data_bound_dudt[0, :, 1] = data[2, :, 1]

    # boundary condition Values
    data[0, :, 2] = 0
    data[1, :, 2] = 0
    data[2, :, 2] = function_bound(data[2, :, 0])
    data_bound_dudt[0, :, 2] = function_dudt_bound(data[2, :, 0])


    data = data.reshape(n_data_per_bc * n_bc, 3)
    data_bound_dudt = data_bound_dudt.reshape(n_data_per_bc * 1, 3)
    #
    x_d, t_d, bound_d = map(lambda x: np.expand_dims(x, axis=1), 
                        [data[:, 0], data[:, 1], data[:, 2]])
    
    x_dudt_d, t_dudt_d, bound_dudt_d = map(lambda x: np.expand_dims(x, axis=1), 
                        [data_bound_dudt[:, 0], data_bound_dudt[:, 1], data_bound_dudt[:, 2]])
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


    if plotting:
        plt.figure(figsize=(5, 5))
        plt.title("Boundary Data points and Collection of points inside")
        plt.scatter(data[:, 0], data[:, 1], marker="x", c=data[:, 2], label="boundary points")
        # plt.scatter(data_bound_dudt[:, 0], 0.5+data_bound_dudt[:, 1], marker="x", c=data_bound_dudt[:, 2], label="boundary points")
        plt.scatter(colloc[:, 0]*Lx, colloc[:, 1]*Lt, s=10, marker="o", c="r", label="inside points")
        plt.scatter(colloc_test[:, 0]*Lx, colloc_test[:, 1]*Lt, s=1, marker="*", c="blue", label="test inside points")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()
        plt.show()


    x_c, t_c, x_d, t_d, bound_d =map(lambda x: torch.tensor(x,dtype=torch.float32).requires_grad_(True),
                                [x_c, t_c, x_d, t_d, bound_d])
    x_dudt_d, t_dudt_d, bound_dudt_d =map(lambda x: torch.tensor(x,dtype=torch.float32).requires_grad_(True),
                                [x_dudt_d, t_dudt_d, bound_dudt_d])
    
    x_t, t_t =map(lambda x: torch.tensor(x,dtype=torch.float32),
                                [x_t, t_t])
    
    start = time.time()
    #for deeper network, increase the number of samples inside the
    num_input, num_output = (2, 1)
    # num_hidden, num_layers = (12, 8) #( 8, 8) #(24, 4) #(64, 6) #(64, 4)  
    pinn = MLP_PINN(num_input, num_output, num_hidden, num_layers)

    early_stopper = EarlyStopper(verbose=False, path='checkpoint.pt', patience=100)

    print("start training: ")
    phys_errs=[]
    test_errs=[]
    parab_pinns = Hyperbolic_1p1_pinns(alpha, Lx, Lt, function_bound, function_dudt_bound,
                                         x_d, t_d, bound_d, x_dudt_d, t_dudt_d, bound_dudt_d, x_c, t_c, 
                                         x_t, t_t, lear_rate, lam_eom=lam_eom, lam_bound=lam_bound, lam_mse=lam_mse,
                                           verbose=verbose)
    parab_pinns.train_Hyperbolic_1p1(pinn, num_iteration, phys_errs, test_errs,
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
    vmax= np.max(exact_solution(X0, T0, function_bound, function_dudt_bound, alpha=alpha, L=Lx))
    im2 = axs[0].pcolormesh(X0, T0, exact_solution(X0, T0, function_bound, function_dudt_bound,
                                                    alpha=alpha, L=Lx), vmax=vmax,  cmap="coolwarm")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("t")
    axs[0].set_title("Exact solution")
    axs[0].set_aspect("equal")

    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im2, cax=cax1)
    # cbar1.set_label('exact solution')

    im1 = axs[1].pcolormesh(X0, T0, S, cmap="coolwarm")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("t")
    axs[1].set_title("PINNs")
    axs[1].set_aspect("equal")

    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    # cbar1.set_label('PINNs')

    vmax= np.max(np.abs(exact_solution(X0, T0, function_bound,
                                        function_dudt_bound, alpha=alpha, L=Lx)-S))
    im3 = axs[2].pcolormesh(X0, T0, exact_solution(X0, T0, function_bound,
                                                    function_dudt_bound, alpha=alpha,
                                                      L=Lx)-S, vmin=-vmax, vmax=vmax, cmap="seismic")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("t")
    axs[2].set_title("Difference")
    axs[2].set_aspect("equal")

    divider1 = make_axes_locatable(axs[2])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im3, cax=cax1)
    # cbar1.set_label('Difference')

    plt.show()





