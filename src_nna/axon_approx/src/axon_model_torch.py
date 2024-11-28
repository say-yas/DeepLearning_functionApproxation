import torch
import torch.nn.functional as F
from torch import nn
from torchsummary import summary
#from torch.utils.data import TensorDataset, DataLoader

from tqdm import tqdm

import numpy as np

class AxonNetwork_torch(nn.Module):
    '''PyTorch model for Axon arhitecture.

    Attributes:
        r: R^{-1} from QR decomposion of [1,x]
        coefs: coefficients for orthogonalization
        norms: coefficients for normalizations
        c: final coefficients for function approximation
        device: device, used for computations, default = 'cpu' 
    '''

    def __init__(self, x, y, basis_coef_init=None, r=None, orth_coefs=None, norms=None, bas_np = None, num_basis_fun=3, device='cpu'):
        '''
        Inits network parameters with precomputed values or randomly

        Args:
            x: input of function (torch.Tensor)
            y: function values (torch.Tensor)
            device: optional, device, used for computations 
            basis_coef_init: optional, list of coefficients for linear combinations forming basis functions
            r: optional, R from QR decomposition of [1, x]
            orth_coefs: optional, coeffitients alpha, obtained from orthogonalization
            norms: optional, norms of basis functions, obtained during basis precomputation
            bas_np: optional, precomputed basis (numpy array)
            num_basis_fun: optional, total number of basis functions for approximation
        '''
        super().__init__()

        # print("xin:", x.shape, "yin:", y.shape)

        layers = []
        self.norms = []
        self.coefs = []
        self.device = device
        if (basis_coef_init is None) or (r is None) or (orth_coefs is None) or (norms is None) or (bas_np is None):
            
            # if there is no given basis -> initialize randomly
            bs = torch.cat([torch.ones((x.shape[0],1)), x], dim=1)
            bs, r = torch.linalg.qr(bs, 'reduced')
            self.r = torch.inverse(r).to(device)
            
            for i in range(num_basis_fun-x.shape[1]-1):
                # layers with random initialization:
                layers.append(nn.Linear(i+x.shape[1]+1, 1, bias=False)) 
            
            self.layers = nn.ModuleList(layers)
            bas = self.build_basis_torch(bs)
            self.c = (bas.t()@y).data.to(device)
        else:
            
            # initialize architecture with precomputed basis:
            bas = torch.from_numpy(bas_np.astype(np.float32))
            self.r = torch.inverse(torch.from_numpy(r.astype(np.float32)))
            self.norms = norms
            self.coefs = [[torch.from_numpy(coef.astype(np.float32)).to(device) for coef in c] for c in orth_coefs]


            for i in range(len(basis_coef_init)):
                layers.append(nn.Linear(i+x.shape[1]+1, 1, bias=False))
                layers[-1].weight = nn.Parameter(torch.from_numpy(basis_coef_init[i].astype(np.float32).reshape(1,-1)), requires_grad=True)
                
            self.layers = nn.ModuleList(layers)
            self.c = (bas.t()@y).data.to(device)
            self.r = self.r.to(device)
    

    def forward(self, x):
        x = self.get_basis_torch(x)
        return x@self.c
   

    def build_basis_torch(self, x): 
        '''
        Basis function initialization: constructs basis from random weights and performs orthonormalization 
        '''
        for l in self.layers:
            
            new_x = F.relu(l(x)) # g(w* v)
            self.coefs.append([(x.t()@new_x).reshape(-1).data.to(self.device)])
            new_x = new_x - (x@x.t()@new_x)
            self.norms.append([torch.norm(new_x).data.to(self.device)])
            new_x = new_x/torch.norm(new_x)

            x = torch.cat([x,new_x], dim=1)
        return x

    def get_basis_torch(self, x):   
        '''
        Inference
        '''
        out = torch.cat([torch.ones((x.shape[0],1)).to(self.device), x], dim=1) # v=[x, 1]
        x = out@self.r # v = R^{-1} v
        
        for i,l in enumerate(self.layers): # from 0 to K-1
            new_x = F.relu(l(x)) # g(w* v)
            for coef, norm in zip(self.coefs[i], self.norms[i]):
                new_x = new_x - (x@coef)[:,None] # vk = g(w* v) - v ak(coef)
                new_x = new_x/norm # vk = vk / betak
            x = torch.cat([x,new_x], dim=1) # v= [v, vk]

        return x


def init_weights_torch(m):
    '''
    Xavier uniform weights initialization
    Args:
        m: PyTorch model
    '''
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


def train_random_model_torch(xs, f, K, num_epochs, num_iters=1000, device='cpu', learning_rate=1e-2):
    '''
    Train model with random initialization
    Args:
        xs: numpy array of points
        f: function to approximate
        K: number of basis functions to add
        num_epochs: number of training epochs 
        device: optional, device, used for computations 
    Returns:
        errors: list of errors
    '''
    fs = f(xs).flatten()

    xs = torch.from_numpy(xs.astype(np.float32)).to(device)
    fs = torch.from_numpy(fs.astype(np.float32)).to(device)

    #loader = DataLoader(TensorDataset(xs, fs), batch_size=xs.shape[0])
    errors = []
    for j in tqdm(range(num_iters)): # train several times
        model = AxonNetwork_torch(xs.cpu(), fs.cpu(), num_basis_fun=K+xs.shape[-1]+1).to(device)
        model.apply(init_weights_torch)
        optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_iters/4), gamma=0.6)

        for i in range(num_epochs):
            # print("xs:", xs.shape, "fs:", fs.shape)
            pred = model(xs)  # full gradient
            loss = F.mse_loss(pred, fs)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        pred = model(xs)

        error = (torch.norm(pred - fs)/ torch.norm(fs)).item()
        errors.append(error)
    return errors