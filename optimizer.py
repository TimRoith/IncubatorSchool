import numpy as np

class pdhg:
    """solve the problem
        f(Kx) + g(x) + h(x)
    using PDHG algorithm
        x_{k+1} = prox_{tau*g}(x_k - tau*(dh(x_k) + Kt*y_k))
        x_r = 2*x_{k+1} - x_k
        y_{k+1} = prox_{sigma*fconj}(y_k + sigma*K*x_r)
    with differentiable h and possibly non-smooth f and g
    """
    def __init__(self, x0, y0, K, prox_fconj, 
                 tau=0.01, sigma=0.01, 
                 n_iter=100, prox_g=None, 
                 dh=None, 
                 energy_fun=None,
                 verbosity=1):
        self.iter = 0
        self.n_iter = n_iter
        self.x = np.copy(x0)
        self.y = y0
        
        self.tau = tau
        self.sigma = sigma
        self.prox_fconj = prox_fconj
        self.K = K
        self.prox_g = prox_g if not prox_g is None else lambda x, tau:x
        self.dh = dh if not dh is None else lambda x:0
        self.energy_fun = energy_fun
        self.verbosity = verbosity
        self.energy_hist = []
    
    def compute(self, verbose=False):
        
        while self.iter < self.n_iter:
            self.iter += 1
            x_prev = self.x
            self.x = self.prox_g(self.x - self.tau*(self.dh(self.x)+self.K.adjoint(self.y)), self.tau)
            self.y = self.prox_fconj(self.y + self.sigma*self.K(2*self.x - x_prev), self.sigma)

            
            if not self.energy_fun is None:
                
                energy = self.energy_fun(self.K, self.x)
                if self.verbosity > 0:
                    print('Iteration ' + str(self.iter) +', energy: ' +str(energy)) 

                self.energy_hist.append(energy)
        return self.x


def compute_primal_res(pd):
    pp = 1/pd.tau * (pd.u_old - pd.u) + theta * A.adjoint(pd.p_old - pd.p)
    return torch.linalg.norm(pp)

def compute_dual_res(pd):
    dd = 1/pd.sigma * (pd.p_old - pd.p) - theta * A(pd.u_old - pd.u)
    return torch.linalg.norm(dd[0]) + torch.linalg.norm(dd[1])

def compute_B(pd, c=0.9):
    B = c/(2 * pd.tau)   *  torch.linalg.norm(pd.u    - pd.u_old)**2 +\
    c/(2 * pd.sigma) * (torch.linalg.norm(pd.p[0] - pd.p_old[0])**2 + torch.linalg.norm(pd.p[1] - pd.p_old[1])**2) -\
    2 * torch.vdot(A.adjoint(pd.p-pd.p_old).ravel(), ((pd.u - pd.u_old)).ravel())
    B = B.real
    return B


class lifted_variable:
    def __init__(self,l):
        self.l = l

    def __len__(self):
        return len(self.l)

    def __add__(self, other):
        other = self._check_allowed_dtypes(other)
        return lifted_variable([self[i] + other[i] for i in range(len(self))])

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._check_allowed_dtypes(other)
        return lifted_variable([self[i] - other[i] for i in range(len(self))])

    def __rsub__(self, other):
        other = self._check_allowed_dtypes(other)
        return lifted_variable([-self[i] + other[i] for i in range(len(self))])

    def __mul__(self, other):
        other = self._check_allowed_dtypes(other)
        return lifted_variable([self[i] * other[i] for i in range(len(self))])

    def __rmul__(self, other):
        other = self._check_allowed_dtypes(other)
        return lifted_variable([other[i] * self[i] for i in range(len(self))])

    def clone(self):
        return lifted_variable([self[i].clone() for i in range(len(self))])

    def __str__(self):
        return str(self.l)

    def __getitem__(self, key):
        return self.l[key]


    def _check_allowed_dtypes(self, other):
        if isinstance(other, (int, float, bool)):
            other = self._promote_scalar(other)
        elif not isinstance(other, lifted_variable):
            raise TypeError("Unsupported operand type(s) for +: " + str(type(self))  + " and " + str(type(other)))

        return other

    def _promote_scalar(self, scalar):
        '''
        Promote a scalar to a lifted_variable of the same length as self.
        This is a lazy implemantation, which is not efficient for large lifted_variables.
        '''
        return lifted_variable([scalar for i in range(len(self))])


def imgrad(u):
    """
    applies a 2D image gradient to the image u of shape (n1,n2)
    
    Parameters
    ----------
    u : numpy 2D array, shape n1, n2
        Image

    Returns
    -------
    (px,py) image gradients in x- and y-directions.

    """
    n1 = u.shape[-2]
    n2 = u.shape[-1]
    px = np.concatenate((u[1:,:]-u[0:-1,:], np.zeros((1,n2))),axis=0)
    py = np.concatenate((u[:,1:]-u[:,0:-1], np.zeros((n1,1))),axis=1)
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
    u1 = np.concatenate((-p[0,0:1,:], -(p[0,1:-1,:]-p[0,0:-2,:]), p[0,-2:-1,:]), axis = 0)
    u2 = np.concatenate((-p[1,:,0:1], -(p[1,:,1:-1]-p[1,:,0:-2]), p[1,:,-2:-1]), axis = 1)
    return -(u1+u2)


class total_variation():
    """
    total variation of 2D image u with shape (n1, n2). Scaled by a constant
    regularization parameter scale. Corresponds to the functional 
        scale * TV(u)
    with u in R^{n1 x n2}
    
    __init__ input:
        - n1, n2:   shape of u
        - scale:    scaling factor, usually a regularization parameter
        
    __call__ input:
        - u:        image of shape n1,n2 or n1*n2,
                    
    TV is computed on a grid via finite differences, assuming equidistant 
    spacing of the grid. The gradient of this potential does not exist since 
    TV is not smooth.
    The proximal mapping is approximated using the dual problem. Pass either 
    a maximum number of steps, an accuracy (in the primal-dual gap), or both 
    to the prox evaluation, for more details see 
        total_variation.inexact_prox
    """
    def __init__(self, n1, n2, scale=1):
        self.n1 = n1
        self.n2 = n2
        self.scale = scale
        
    def _imgrad(self, u):
        """
        applies a 2D image gradient to the image u of shape (n1,n2)
        
        Parameters
        ----------
        u : numpy 2D array, shape n1, n2
            Image

        Returns
        -------
        (px,py) image gradients in x- and y-directions.

        """
        px = np.concatenate((u[1:,:]-u[0:-1,:], np.zeros((1,self.n2))),axis=0)
        py = np.concatenate((u[:,1:]-u[:,0:-1], np.zeros((self.n1,1))),axis=1)
        return np.concatenate((px[np.newaxis,:,:],py[np.newaxis,:,:]), axis=0)
    
    def _imdiv(self, p):
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
        u1 = np.concatenate((-p[0,0:1,:], -(p[0,1:-1,:]-p[0,0:-2,:]), p[0,-2:-1,:]), axis = 0)
        u2 = np.concatenate((-p[1,:,0:1], -(p[1,:,1:-1]-p[1,:,0:-2]), p[1,:,-2:-1]), axis = 1)
        return u1+u2
    
    def __call__(self, u):
        """
        Computes the TV-seminorm of u
        
        Parameters 
        ----------
        u : numpy array of shape n1, n2
        
        Returns
        -------
        TV(u) (scalar)
        """
        return self.scale * np.sum(np.sqrt(np.sum(self._imgrad(u)**2,axis=0)))