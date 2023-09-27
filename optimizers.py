import numpy as np
from operators import Grad, soft_shrinkage
import pywt

class optimizer:
    def __init__(self, max_it=10, verbosity=1, energy_fun=None):
        self.max_it = max_it
        self.num_it = 0
        self.verbosity = verbosity
        self.energy_fun=energy_fun
        self.cur_energy = float('inf')
        self.energy_hist = []
        
    def solve(self):
        while not self.terminate():
            self.step()
            self.num_it+=1
            if not self.energy_fun is None:
                energy = self.energy_fun(self.x)
                self.cur_energy=energy
                self.energy_hist.append(energy)
            if self.verbosity > 0:
                print('Iteration: ' + str(self.num_it))
                print('Energy: ' +str(self.cur_energy))
                
        # return solution
        return self.x
      
    
    def terminate(self):
        if self.num_it > self.max_it:
            return True
        else:
            return False

class split_Bregman_TV(optimizer):
    def __init__(self, A, rhs,x0, 
                 gamma=1.0, lamda=1.0,
                 inner_verbosity = 0,
                 max_inner_it = 10,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.A = A
        self.gamma = gamma
        self.lamda = lamda
        self.grad = Grad()

        
        class cg_op:
            def __init__(self,):
                self.grad = Grad()
            
            
            def __call__(self,x):
                return lv([gamma * A(x), 0.5 * self.grad(x)])

            def adjoint(self, p):
                return gamma * A.adjoint(p[0]) + 0.5 * self.grad.adjoint(p[1])
            
        self.cg_op = cg_op()
        self.rhs = rhs

        self.x = x0
        self.y = self.grad(x0)
        self.d = soft_shrinkage(self.y + self.grad(self.x), self.lamda * self.gamma)
        self.inner_verbosity = inner_verbosity
        self.max_inner_it = max_inner_it

    def step(self):
        inner_rhs = lv([self.gamma * self.rhs, 0.5 * (self.d - self.y)])
        self.x = self.solve_inner(inner_rhs)
        self.d = soft_shrinkage(self.y + self.grad(self.x), self.lamda * self.gamma)
        self.y = self.y + self.grad(self.x) - self.d
        
    def solve_inner(self, rhs):
        return lscg(self.cg_op, rhs, self.x, 
                    verbosity = self.inner_verbosity, 
                    max_it=self.max_inner_it).solve()
    
    

class ista_L1(optimizer):
    def __init__(self, A, x, y, t=0.1, lamda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.x = x
        self.y = y
        self.lamda = lamda
        self.t = t
    
    def step(self,):
        grad = self.A.adjoint(self.A(self.x) - self.y)
        lin_up = self.x - 2 * self.t * grad
        self.x = soft_shrinkage(lin_up, self.lamda * self.t)

class ista_wavelets(optimizer):
    def __init__(self, A, x, y, wave, t=0.1, lamda=1.0, **kwargs):
        super().__init__(**kwargs)

        def energy_fun(x):
            coeffs,_ = pywt.coeffs_to_array(pywt.wavedec2(x, wavelet=wave, mode='periodization'))
            return 0.5* np.linalg.norm(A(x)-y)**2 + lamda* np.linalg.norm(coeffs, ord=1)

        self.A = A
        self.x = x
        self.y = y
        self.lamda = lamda
        self.t = t
        self.wave = wave
        self.energy_fun = energy_fun
    
    def step(self,):
        grad = self.A.adjoint(self.A(self.x) - self.y)
        lin_up = self.x - 2 * self.t * grad
        lin_up_coeffs, slices = pywt.coeffs_to_array(pywt.wavedec2(lin_up, wavelet=self.wave, mode='periodization'))
        x_coeffs = soft_shrinkage(lin_up_coeffs, self.lamda * self.t)
        self.x = pywt.waverec2(pywt.array_to_coeffs(x_coeffs, slices, output_format='wavedec2'), wavelet=self.wave, mode='periodization')




class lscg(optimizer):
    '''
    Solve the linear system Ax=y using the conjugate gradient method

    This implements Algorithm 1 from [1], or Algorithm 3 from [2]

    References:
    ----------
    [1] Fast Conjugate Gradient Algorithms with Preconditioning for Compressed Sensing
        J. A. Tropp, A. C. Gilbert, M. A. Saunders
        IEEE Transactions on Information Theory, 2007

    [2] https://sites.stat.washington.edu/wxs/Stat538-w03/conjugate-gradients.pdf
    '''
    def __init__(self, A, y, x, **kwargs):
        super().__init__(**kwargs)
        self.A = A
        self.y = y
        self.x = x.copy()
        self.r = y - A(x)

        self.g = self.A.adjoint(self.r)
        self.p = self.g
        


    def step(self):
        #self.gg = self.g.norm()**2
        self.gg = np.linalg.norm(self.g)**2
        if self.verbosity > 0:
            print('Iteration ' + str(self.num_it) + ', norm of g: ' + str(self.gg))

        if self.num_it > 0:
            self.beta = -self.gg / self.gg_old
            self.p = self.g - self.beta * self.p

        Ap = self.A(self.p)
        alpha = float(self.gg / Ap.norm()**2)
        self.x = self.x + alpha * self.p
        self.r = self.r - alpha * Ap
        self.g = self.A.adjoint(self.r)
        self.gg_old = self.gg




class lv:
    def __init__(self,l):
        self.l = l

    def __len__(self):
        return len(self.l)

    def __add__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([self[i] + other[i] for i in range(len(self))])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([self[i] - other[i] for i in range(len(self))])

    def __rsub__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([-self[i] + other[i] for i in range(len(self))])

    def __mul__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([self[i] * other[i] for i in range(len(self))])

    def __rmul__(self, other):
        other = self._check_allowed_dtypes(other)
        return lv([other[i] * self[i] for i in range(len(self))])

    def clone(self):
        return lv([self[i].clone() for i in range(len(self))])

    def norm(self):
        return np.sqrt(sum([np.linalg.norm(self[i])**2 for i in range(len(self))]))

    def __str__(self):
        return str(self.l)

    def __getitem__(self, key):
        return self.l[key]


    def _check_allowed_dtypes(self, other):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        elif not isinstance(other, lv):
            raise TypeError("Unsupported operand type(s) for +: " + str(type(self))  + " and " + str(type(other)))

        return other

    def _promote_scalar(self, scalar):
        '''
        Promote a scalar to a lv of the same length as self.
        This is a lazy implemantation, which is not efficient for large lvs.
        '''
        return lv([scalar for i in range(len(self))])





# class total_variation():
#     """
#     total variation of 2D image u with shape (n1, n2). Scaled by a constant
#     regularization parameter scale. Corresponds to the functional 
#         scale * TV(u)
#     with u in R^{n1 x n2}
    
#     __init__ input:
#         - n1, n2:   shape of u
#         - scale:    scaling factor, usually a regularization parameter
        
#     __call__ input:
#         - u:        image of shape n1,n2 or n1*n2,
                    
#     TV is computed on a grid via finite differences, assuming equidistant 
#     spacing of the grid. The gradient of this potential does not exist since 
#     TV is not smooth.
#     The proximal mapping is approximated using the dual problem. Pass either 
#     a maximum number of steps, an accuracy (in the primal-dual gap), or both 
#     to the prox evaluation, for more details see 
#         total_variation.inexact_prox
#     """
#     def __init__(self, n1, n2, scale=1):
#         self.n1 = n1
#         self.n2 = n2
#         self.scale = scale
        
#     def _imgrad(self, u):
#         """
#         applies a 2D image gradient to the image u of shape (n1,n2)
        
#         Parameters
#         ----------
#         u : numpy 2D array, shape n1, n2
#             Image

#         Returns
#         -------
#         (px,py) image gradients in x- and y-directions.

#         """
#         px = np.concatenate((u[1:,:]-u[0:-1,:], np.zeros((1,self.n2))),axis=0)
#         py = np.concatenate((u[:,1:]-u[:,0:-1], np.zeros((self.n1,1))),axis=1)
#         return np.concatenate((px[np.newaxis,:,:],py[np.newaxis,:,:]), axis=0)
    
#     def _imdiv(self, p):
#         """
#         Computes the negative divergence of the 2D vector field px,py.
#         can also be seen as a tensor from R^(n1xn2x2) to R^(n1xn2)

#         Parameters
#         ----------
#             - p : 2 x n1 x n2 np.array

#         Returns
#         -------
#             - divergence, n1 x n2 np.array
#         """
#         u1 = np.concatenate((-p[0,0:1,:], -(p[0,1:-1,:]-p[0,0:-2,:]), p[0,-2:-1,:]), axis = 0)
#         u2 = np.concatenate((-p[1,:,0:1], -(p[1,:,1:-1]-p[1,:,0:-2]), p[1,:,-2:-1]), axis = 1)
#         return u1+u2
    
#     def __call__(self, u):
#         """
#         Computes the TV-seminorm of u
        
#         Parameters 
#         ----------
#         u : numpy array of shape n1, n2
        
#         Returns
#         -------
#         TV(u) (scalar)
#         """
#         return self.scale * np.sum(np.sqrt(np.sum(self._imgrad(u)**2,axis=0)))
    
    
    
def imgrad(x):
    """
    applies a 2D image gradient to the image x of shape (n1,n2)
    
    Parameters
    ----------
    x : numpy 2D array, shape n1, n2
        Image

    Returns
    -------
    (px,py) image gradients in x- and y-directions.

    """
    n1 = x.shape[-2]
    n2 = x.shape[-1]
    px = np.concatenate((x[1:,:]-x[0:-1,:], np.zeros((1,n2))),axis=0)
    py = np.concatenate((x[:,1:]-x[:,0:-1], np.zeros((n1,1))),axis=1)
    return np.concatenate((px[None,...],py[None,...]), axis=0)

def imdiv(p):
    """
    Computes the negative divergence of the 2D vector field px,py.
    can also be seen as a tensor from R^(n1xn2x2) to R^(n1xn2)

    Parameters
    ----------
        - p : 2 x n1 x n2 np.array

    Returns
    -------
        - divergence, n1 x n2 np.array
    """
    x1 = np.concatenate((-p[0,0:1,:], -(p[0,1:-1,:]-p[0,0:-2,:]), p[0,-2:-1,:]), axis = 0)
    x2 = np.concatenate((-p[1,:,0:1], -(p[1,:,1:-1]-p[1,:,0:-2]), p[1,:,-2:-1]), axis = 1)
    return -(x1+x2)
    

# class pdhg:
#     """solve the problem
#         f(Kx) + g(x) + h(x)
#     using PDHG algorithm
#         x_{k+1} = prox_{tau*g}(x_k - tau*(dh(x_k) + Kt*y_k))
#         x_r = 2*x_{k+1} - x_k
#         y_{k+1} = prox_{sigma*fconj}(y_k + sigma*K*x_r)
#     with differentiable h and possibly non-smooth f and g
#     """
#     def __init__(self, x0, y0, K, prox_fconj, 
#                  tau=0.01, sigma=0.01, 
#                  n_iter=100, prox_g=None, 
#                  dh=None, 
#                  energy_fun=None,
#                  verbosity=1):
#         self.iter = 0
#         self.n_iter = n_iter
#         self.x = np.copy(x0)
#         self.y = y0
        
#         self.tau = tau
#         self.sigma = sigma
#         self.prox_fconj = prox_fconj
#         self.K = K
#         self.prox_g = prox_g if not prox_g is None else lambda x, tau:x
#         self.dh = dh if not dh is None else lambda x:0
#         self.energy_fun = energy_fun
#         self.verbosity = verbosity
#         self.energy_hist = []
    
#     def compute(self, verbose=False):
        
#         while self.iter < self.n_iter:
#             self.iter += 1
#             x_prev = self.x
#             self.x = self.prox_g(self.x - self.tau*(self.dh(self.x)+self.K.adjoint(self.y)), self.tau)
#             self.y = self.prox_fconj(self.y + self.sigma*self.K(2*self.x - x_prev), self.sigma)

            
#             if not self.energy_fun is None:
                
#                 energy = self.energy_fun(self.K, self.x)
#                 if self.verbosity > 0:
#                     print('Iteration ' + str(self.iter) +', energy: ' +str(energy)) 

#                 self.energy_hist.append(energy)
#         return self.x