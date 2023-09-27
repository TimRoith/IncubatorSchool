from optimizer import lscg
from optimizer import imgrad, imdiv
from optimizer import lv
from operators import Grad, soft_shrinkage, Radon
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np

grad = Grad()
#%%
num_theta = 150
dim = 150
noise_lvl = 0.01 * dim


phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
phantom = ski.transform.resize(phantom, (dim, dim))
theta = np.linspace(0,180, endpoint = False, num=num_theta)#
R = Radon(theta=theta)


sinogram =  R(phantom) # ski.transform.radon(phantom, theta)
sinogram += np.random.normal(0, noise_lvl, size=sinogram.shape)

# def soft_shrinkage(x, lamda):
#     return np.maximum(np.abs(x)-lamda, 0.) * np.sign(x)

class Radon:
    def __call__(self, u):
        return ski.transform.radon(u, theta)*(np.pi/(2 * num_theta))
    
    def adjoint(self, k):
        return ski.transform.iradon(k, theta, filter_name=None)

class Grad:
    def __call__(self, u):
        return grad(u)

    def adjoint(self, p):
        return -imdiv(p)
    
def energy_fun(A, u, lamda=1.):
    return 1/2 * np.linalg.norm(A(u) - sinogram)**2 + lamda * np.sum(np.abs(grad(u)))
    
x0 = ski.transform.iradon(sinogram, theta=theta)

class split_Bregman_TV:
    def __init__(self, A, rhs, u0, 
                 gamma=1.0, lamda=1.0,
                 max_it=10, verbosity = 1,
                 energy_fun = None):
        self.A = A
        self.gamma = gamma
        self.lamda = lamda
        self.num_it = 0
        self.max_it = max_it
        self.verbosity = verbosity
        self.energy_fun = energy_fun
        self.cur_energy = float('inf')
        self.energy_hist = []
        
        class cg_op:
            def __call__(self, u):
                return lv([gamma * A(u), 0.5 * grad(u)])

            def adjoint(self, p):
                return gamma * A.adjoint(p[0]) + 0.5 * grad.adjoint(p[1])

        self.cg_op = cg_op()
        self.rhs = rhs

        self.u = u0
        self.b = grad(u0)
        self.d = soft_shrinkage(self.b + grad(self.u), self.lamda * self.gamma)

    def step(self):
        cg_rhs = lv([self.gamma * self.rhs, 0.5 * (self.d - self.b)])
        self.u = lscg(self.cg_op, cg_rhs, self.u).solve()
        self.d = soft_shrinkage(self.b + grad(self.u), self.lamda * self.gamma)
        self.b = self.b + grad(self.u) - self.d
        self.num_it += 1
        
        if not self.energy_fun is None:
            energy = self.energy_fun(self.A, self.u, lamda=self.lamda)
            self.cur_energy=energy
            self.energy_hist.append(energy)

    def terminate(self):
        if self.num_it > self.max_it:
            return True
        else:
            return False

    def solve(self,):
        while not self.terminate():
            self.step()
            if self.verbosity > 0:
                print('Iteration: ' + str(self.num_it))
                print('Energy: ' +str(self.cur_energy))
                
#%%
gamma = 1
sBTV = split_Bregman_TV(Radon(), sinogram, x0, gamma=gamma, 
                        energy_fun=energy_fun, lamda = 1.)
sBTV.solve()

#%%
plt.close('all')
plt.imshow(sBTV.u)
plt.figure()
plt.imshow(x0)
