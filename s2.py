from optimizer import split_Bregman_TV
from operators import Radon, TV
import matplotlib.pyplot as plt
import skimage as ski
import numpy as np

#%%
num_theta = 90
dim = 100
noise_lvl = 0.005
phantom = ski.img_as_float(ski.data.shepp_logan_phantom())
phantom = ski.transform.resize(phantom, (dim, dim))


theta = np.linspace(0,180, endpoint = False, num=num_theta)
R = Radon(theta=theta)

sinogram =  R(phantom)
sinogram += np.random.normal(0, noise_lvl, size=sinogram.shape)

#%%
u0 = R.inverse(sinogram)

def energy_fun(A, u, lamda=1.):
    return 1/2 * np.linalg.norm(A(u) - sinogram)**2 + lamda * TV()(u)

sBTV = split_Bregman_TV(R, sinogram, u0, gamma=1.0, 
                        energy_fun=energy_fun, lamda = .02,
                        max_inner_it = 2)
sBTV.solve()

#%%
plt.close('all')
plt.imshow(sBTV.u)
plt.figure()
plt.imshow(u0)