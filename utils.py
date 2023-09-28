import skimage as ski
import matplotlib.pyplot as plt
from skimage.draw import disk, polygon, ellipse
import numpy as np

def get_phantom(dim):
    phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
    return ski.transform.resize(phantom, (dim, dim))

def get_DESY(dim):
    I = plt.imread('DESY.png').sum(axis=-1)
    I[np.where(I>0.)]=1.
    
    return ski.transform.resize(I, (dim, dim))


class shapes:
    def __init__(self, img_size=100, p=0., noise_lvl=0.):
        self.img_size = [img_size, img_size]
        self.DESY = get_DESY(img_size)
        self.p = p
        self.noise_lvl = noise_lvl
        self.shape_names = ['rectangle', 'circle', 'triangle', 'ellipse']
        
    def get_shape(self, name='rectangle'):
        if name == 'rectangle':
            I =  self.rectangle()
        elif name == 'circle':
            I = self.circle()
        elif name == 'triangle':
            I = self.triangle()
        elif name == 'ellipse':
            I = self.ellipse()
        elif name == 'random':
            i = np.random.randint(0, len(self.shape_names)-1)
            shape_name = self.shape_names[i]
            I = self.get_shape(name=shape_name)
        else:
            raise RuntimeError('Unknwon shape: ' + str(name))
            
        if np.random.binomial(1,self.p):
            I = np.maximum(self.DESY, I)
            
        return I

    def rectangle(self,):
        S = np.zeros(self.img_size)
        x_start, y_start = np.random.randint(self.img_size[1]//4,self.img_size[1]//3 , size=(2,))
        x_end, y_end = np.random.randint(self.img_size[1]//2, 3*self.img_size[1]//4, size=(2,))
        S[x_start:x_end, y_start:y_end] = 1
        return S
        
    def circle(self,):
        C = np.zeros(self.img_size)
        radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)
        row = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        col = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        # modern scikit uses a tuple for center
        rr, cc = disk((row, col), radius)
        C[rr, cc] = 1.
        return C
    
    def triangle(self,):
        T = np.zeros(self.img_size)
        
        poly = np.random.randint(self.img_size[0]//4, 3 * self.img_size[0]//4, (3,2))
        rr, cc = polygon(poly[:, 0], poly[:, 1], T.shape)
        T[rr, cc] = 1
        return T
    
    def ellipse(self,):
        E = np.zeros(self.img_size)
        r_radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)
        c_radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)

        row = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        col = self.img_size[1]//2# + np.random.randint(-self.img_size[1]//6, self.img_size[1]//6)
        
        rot = np.random.uniform(-np.pi, np.pi)
        
        rr, cc = ellipse(row, col, r_radius, c_radius, shape=None, rotation=rot)
        E[rr, cc] = 1.
        return E