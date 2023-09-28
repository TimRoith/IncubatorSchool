import numpy as np
import skimage as ski

def soft_shrinkage(x, lamda):
    return np.maximum(np.abs(x)-lamda, 0.) * np.sign(x)

class Identity:
    def __call__(self,u):
        return u
    
    def adjoint(self, u):
        return u
    
    def inv(self, u):
        return u

class Radon:
    def __init__(self, theta=None):
        self.theta = theta if not theta is None else np.linspace(0,180, 50)
        self.num_theta = len(self.theta)
    
    def __call__(self, u):
        return ski.transform.radon(u, self.theta)/u.shape[-1]
    
    def adjoint(self, k):
        return ski.transform.iradon(k, self.theta, filter_name=None)/(k.shape[0] * np.pi/(2 * self.num_theta))
    
    def inv(self, k):
        return ski.transform.iradon(k * k.shape[0], self.theta)
    
    inverse = inv
    T = adjoint

class TV:
    def __init__(self,):
        self.grad = Grad()
    
    def __call__(self, u):
        return np.linalg.norm(self.grad(u).ravel(), ord=1)
    
def test_adjoint(A, x, y=None):
    Ax = A(x)
    if y is None:
        y = np.random.uniform(size=Ax.shape)
    res_1 = np.sum(Ax * y)
    res_2 = np.sum(x * A.adjoint(y))
    return res_1, res_2

class L1_norm:
    def __call__(self, u, lamda=1.):
        return lamda * np.linalg.norm(u.ravel(), ord=1)

    def prox(self, u, lamda=1.):
        return soft_shrinkage(u, lamda)



class Grad:
    def __call__(self, u):
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

    def adjoint(self, p):
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
        return (u1+u2)