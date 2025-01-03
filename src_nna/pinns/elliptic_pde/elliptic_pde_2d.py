import torch
import time
import numpy as np
from scipy.stats import qmc

from  src_nna.pinns.nn_pinns import MLP_PINN, EarlyStopper
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tqdm import tqdm
import matplotlib.pyplot as plt

torch.manual_seed(123)

def cos(x, constant=3):
    """
    cosine function to be used in the boundary condition
    """
    return constant * np.cos(x)

def sin(x, constant=3):
    """
    sine function to be used in the boundary condition
    """
    return constant * np.sin(x)

def x_sqrt(x, constant=3):
    """
    square root function to be used in the boundary condition
    """
    return constant * np.sqrt(x)

def x_pow2(x, constant=3):
    """
    power 2 function to be used in the boundary condition
    """
    return constant * np.power(x,2)

def x_lin(x, constant=3):
    """
    linear function to be used in the boundary condition
    """
    return constant * x

def cnst(x, constant=3):
    """
    constant function to be used in the boundary condition
    """
    return constant

def func_in( x,  n, func):
    return func(x)*np.sin(n*np.pi*x)


def exact_solution(x, y, func, L=1.0):
    """
    exact solution of the Elliptic 1+1 space-time dimensional PDE
    u(x, y) = sum_{n=1}^{\infty} a_n sinh(n*pi*y) sin(n*pi*x)
    """
    def compute_an(func, L, n):
        from  scipy import integrate 
        res = integrate.quad( func_in, 0, L, args=(n, func))
        return 2.0 * res[0]

    utx = 0
    for n in np.arange(1, 10):
        an = compute_an(func, L, n) 
        fac = an / np.sinh(n*np.pi )
        utx += fac * np.sinh(n*np.pi*y)  * np.sin(n*np.pi*x)
    return  utx


class Elliptic_2d_pinns:
    """
    Class to train the PINN model to solve the 2D heat equation
    Parameters:
    Lx: float, the length of the x domain
    Ly: float, the length of the y domain
    func: function, the function to be used in the boundary condition
    x_d: torch.tensor, the x component of the boundary condition
    y_d: torch.tensor, the y component of the boundary condition

    bound_d: torch.tensor, the boundary condition values
    x: torch.tensor, the x component of the points inside the domain
    y: torch.tensor, the y component of the points inside the domain
    x_test: torch.tensor, the x component of the test points inside the domain
    y_test: torch.tensor, the y component of the test points inside the domain
    lear_rate: float, the learning rate
    lam_eom: float, the weight of the equation of motion loss
    lam_bound: float, the weight of the boundary condition loss
    lam_mse: float, the weight of the mean square error loss
    verbose: bool, print the loss function values
    """
    def __init__(self,  Lx, Ly, func,  x_d, y_d, bound_d, x, y,
                  x_test, y_test, lear_rate=0.001, lam_eom=1.0, lam_bound=1.0, lam_mse=1.0 , verbose=False):
        self. Lx = Lx
        self.Ly =Ly

        self.func = func

        self.lear_rate = lear_rate
        self.lam_eom, self.lam_bound, self.lam_mse = lam_eom, lam_bound, lam_mse 

        self.x_d, self.y_d, self.bound_d, self.x, self.y =  x_d, y_d, bound_d, x, y

        self.x_test, self.y_test = x_test, y_test

        self.verbose = verbose



    def loss_Elliptic_2d(self, model):
        """
        loss function for the 2D heat equation
        loss = loss_boundary + loss_physics + loss_MSE
        params:
        model: torch model, the model to be trained
        """

        # boundary loss
        xy = torch.cat([self.x_d, self.y_d], dim=1)
        u_b = model(xy)
        loss_bound = torch.mean((self.bound_d - u_b)**2)

        # Equation of motion loss and mse exact sol
        u_mse = model(torch.cat([self.x, self.y], dim=1))
        # loss exact solution
        ex_sol = exact_solution(self.x.detach().numpy(), self.y.detach().numpy(), self.func, self.Lx)
        loss_mse = torch.mean((torch.tensor(ex_sol) - u_mse)**2)


        # Equation of motion loss and mse exact sol
        u = model(torch.cat([self.x, self.y], dim=1))
        # Compute first-order gradients
        #torch.ones_like(u),or remove it and keep u.sum() to get the sum of u
        u_x = torch.autograd.grad(u, self.x, torch.ones_like(u),  create_graph=True)[0] 
        u_y = torch.autograd.grad(u, self.y, torch.ones_like(u), create_graph=True)[0]
        
        # Compute second-order gradients
        u_xx = torch.autograd.grad(u_x, self.x, torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, self.y, torch.ones_like(u_y), create_graph=True)[0]
        
        # Compute differential operator
        F =  u_yy + u_xx 

        loss_eom = torch.mean(F**2) # EOM loss


        loss = self.lam_eom* loss_eom+ self.lam_bound * loss_bound + self.lam_mse * loss_mse
        if self.verbose:
            print(loss_eom, loss_bound, loss_mse)
        # Return total loss
        return loss
    
    def train_Elliptic_2d(self, model, num_iter, phys_errors, test_errors, early_stopper=None, plotting=False):  
        
        """
        train the PINN model to solve the 1_1 Elliptic equation
        params:
        model: torch model, the model to be trained
        num_iter: int, the number of iterations
        phys_errors: list, the physical errors
        test_errors: list, the validation errors
        early_stopper: EarlyStopper, the early stopping object
        plotting: bool, plot the loss functions
        """

        optimizer = torch.optim.Adam(model.parameters(),lr= self.lear_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                                factor=0.9, patience=40,
                                                                threshold=1e-8)
        lossess = []
        for i in tqdm(range(num_iter+1)):
            optimizer.zero_grad()
            model.train()
            
            # compute each term of the PINN loss function
            loss = self.loss_Elliptic_2d(model)
            lossess.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            
            xy = torch.cat([self.x, self.y], dim=1)
            u = model(xy)
            exact_sols_phys = exact_solution(self.x.detach().numpy(), self.y.detach().numpy(),
                                              self.func, self.Lx)
            phys_err = torch.mean((u-torch.tensor(exact_sols_phys))**2)
            phys_errors.append(phys_err.detach().numpy())

            model.eval()
            # Disable gradient computation and reduce memory consumption.
            with torch.no_grad():
                xy_test= torch.cat([self.x_test, self.y_test], dim=1)
                u_t = model(xy_test)
                exact_sols_phys = exact_solution(self.x_test.detach().numpy(), self.y_test.detach().numpy(),
                                              self.func, self.Lx)
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
        
            # plot the resuLy as training progresses
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


def run_traning(n_bc = 3, Lx=1.0, Ly=0.7, Nc = 700 , n = 700, num_hidden=32, num_layers=8,
                 lear_rate= 5e-3, num_iteration = 1600, function_bound = cos,
                   n_data_per_bc = 2000, plotting = False, verbose=False):
    """
    run the training of the PINN model to solve the 2D heat equation
    params:
    n_bc: int, the number of boundary conditions
    Lx: float, the length of the x domain
    Ly: float, the length of the y domain
    Nc: int, the number of points inside the domain
    n: int, the number of points to plot the results
    num_hidden: int, the number of hidden units
    num_layers: int, the number of hidden layers
    lear_rate: float, the learning rate
    num_iteration: int, the number of iterations
    function_bound: function, the function to be used in the boundary condition
    n_data_per_bc: int, the number of data points per boundary condition
    plotting: bool, plot the results
    verbose: bool, print the loss function values
    """

    params=[Lx, Ly]
    #
    engine = qmc.LatinHypercube(d=1)
    data = np.zeros([n_bc, n_data_per_bc, 3])

    for i, j in zip(range(n_bc), [0, Lx, 0, Ly]):
        points = (engine.random(n=n_data_per_bc)[:, 0] ) 
        if i < 2:
            data[i, :, 0] = j #x component, x boundary 
            data[i, :, 1] = points*Ly #y component, x boundary 
        else:
            data[i, :, 0] = points*Lx #x component, y boundary 
            data[i, :, 1] = j #y component, y boundary 

    # boundary condition Values
    data[0, :, 2] = 0
    data[1, :, 2] = 0
    data[2, :, 2] = 0
    data[3, :, 2] = function_bound(data[3, :, 0])

    data = data.reshape(n_data_per_bc * n_bc, 3)
    #
    x_d, y_d, bound_d = map(lambda x: np.expand_dims(x, axis=1), 
                        [data[:, 0], data[:, 1], data[:, 2]])

#
    # sampling from points inside the square
    engine = qmc.LatinHypercube(d=2)
    colloc = engine.random(n=Nc)
    colloc_test = engine.random(n=2*Nc)
    #
    x_c, y_c = map(lambda x: np.expand_dims(x, axis=1), 
                [colloc[:, 0]*Lx, colloc[:, 1]*Ly])
    x_t, y_t = map(lambda x: np.expand_dims(x, axis=1), 
                [colloc_test[:, 0]*Lx, colloc_test[:, 1] *Ly])

    if plotting:
        plt.figure(figsize=(5, 5))
        plt.title("Boundary Data points and Collection of points inside")
        plt.scatter(colloc[:, 0]*Lx, colloc[:, 1]*Ly, s=10, marker="o", c="r", label="inside points")
        plt.scatter(colloc_test[:, 0]*Lx, colloc_test[:, 1]*Ly, s=1, marker="*", c="blue", label="test inside points")
        plt.scatter(data[:, 0], data[:, 1], marker="x", c=data[:, 2],  label="boundary points")
        plt.xlabel("x")
        plt.ylabel("t")
        plt.colorbar()
        plt.show()



    x_c, y_c, x_d, y_d, bound_d =map(lambda x: torch.tensor(x,dtype=torch.float32).requires_grad_(True),
                                [x_c, y_c, x_d, y_d, bound_d])
    
    x_t, y_t =map(lambda x: torch.tensor(x,dtype=torch.float32),
                                [x_t, y_t])
    
    start = time.time()
    #for deeper network, increase the number of samples inside the
    num_input, num_output = (2, 1)
    # num_hidden, num_layers = (16, 6)#( 2**7, 6) #( 32, 8) #(24, 4) #(10, 6) #(64, 6) #(64, 4)  
    pinn = MLP_PINN(num_input, num_output, num_hidden, num_layers)

    early_stopper = EarlyStopper(verbose=False, path='checkpoint.pt', patience=100)

    print("start training: ")
    phys_errs=[]
    test_errs=[]
    parab_pinns = Elliptic_2d_pinns(Lx, Ly, function_bound,  x_d, y_d, bound_d, x_c, y_c,
                  x_t, y_t, lear_rate, lam_eom=1.0, lam_bound=0.5, lam_mse=1.0, verbose=verbose)
    parab_pinns.train_Elliptic_2d(pinn, num_iteration, phys_errs, test_errs,
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
    Y = np.linspace(0, Ly, n)
    X0, Y0 = np.meshgrid(X, Y)
    X = X0.reshape([n*n, 1])
    Y = Y0.reshape([n*n, 1])
    X_T = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)
    xy = torch.cat([X_T, Y_t], dim=1)
    S = pinn(xy)
    S = S.detach().numpy().reshape(n, n)

    #
    fig, axs = plt.subplots(1, 3, figsize=(13, 2.6))
    vmax= np.max(exact_solution(X0, Y0, function_bound,  L=Lx))
    im2 = axs[0].pcolormesh(X0, Y0, exact_solution(X0, Y0, function_bound,  L=Lx),  vmax=vmax, cmap="coolwarm")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_title("Exact solution")
    axs[0].set_aspect("equal")

    divider1 = make_axes_locatable(axs[0])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im2, cax=cax1)
    # cbar1.set_label('exact solution')

    im1 = axs[1].pcolormesh(X0, Y0, S,vmax=vmax, cmap="coolwarm")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    axs[1].set_title("PINNs")
    axs[1].set_aspect("equal")

    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im1, cax=cax1)
    # cbar1.set_label('PINNs')

    vmax= np.max(np.abs(exact_solution(X0, Y0, function_bound, L=Lx)-S))
    im3 = axs[2].pcolormesh(X0, Y0, exact_solution(X0, Y0, function_bound, L=Lx)-S, vmax=vmax, cmap="seismic")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title("Difference")
    axs[2].set_aspect("equal")

    divider1 = make_axes_locatable(axs[2])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    cbar1 = plt.colorbar(im3, cax=cax1)
    # cbar1.set_label('Difference')

    plt.show()
