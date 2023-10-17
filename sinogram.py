import skimage as ski
import numpy as np
import operators
import matplotlib.pyplot as plt
#%%
phantom = ski.img_as_float(ski.data.shepp_logan_phantom())

num_angles = 50
N = phantom.shape[0]
theta = np.linspace(0,180, endpoint = False, num=num_angles)
R = operators.Radon(theta)
sinogram =  R(phantom)
ray_pos = np.arange(20, 2*phantom.shape[0], step=40)
rays = np.zeros([2*N,2*N])
crop = phantom.shape[0]//2

rays[:,ray_pos] = 0.5
rays[:,ray_pos+1] = 1
rays[:,ray_pos+2] = 0.5


angles = np.linspace(0, np.pi, num_angles)

#%%
plt.close('all')
fig, axs = plt.subplots(1, 3, figsize=(20,5))


for i,angle in enumerate(angles):
    
    deg = 180 * angle/np.pi
    
    #rot = ski.transform.rotate(phantom, deg)
    rot = ski.transform.rotate(rays, deg)
    rot = rot[crop:-crop,:][:,crop:-crop]
    
    rot = np.maximum(rot, phantom)
 
    projection = sinogram[:,i]
    sino = sinogram[:,:i+1]
    axs[0].imshow(rot, cmap = 'gray');
    axs[0].tick_params(bottom = False, left = False)
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title('Source')
    axs[0].set_xlabel('Detector')
    axs[1].clear()
    axs[1].plot(np.arange(phantom.shape[0]), projection, linewidth=3., color='xkcd:sky')
    
    axs[1].set_ylim([0,0.4]);
    axs[1].set_title('Measurement at '+str(deg)+'$^{\circ}$')
    axs[1].set_xlabel('position')
    axs[1].set_ylabel('intensity decay')
    axs[2].imshow(sino, cmap = 'gray')
    axs[2].axis('auto')
    axs[2].set_xlim([0,num_angles-1])
    axs[2].set_title('Sinogram')
    #axs[2].set_xticks([0,100,200,300,400], ['$0^{\circ}$', '$45^{\circ}$', '$90^{\circ}$', '$135^{\circ}$', '$180^{\circ}$'])
    axs[2].set_xlabel('angle')
    axs[2].set_ylabel('position')
    
    plt.show()
    plt.pause(1)
    plt.tight_layout(pad=0.0)
    plt.savefig('ct/' +str(i) + '.png')
#%% 

ray_pos = np.arange(20, phantom.shape[0], step=40)
rays = np.zeros([phantom.shape[0],phantom.shape[0],4])
rays[:,ray_pos,:] = 0.75
rays[:,ray_pos+1,:] = 1
rays[:,ray_pos+2,:] = 0.75
plt.imshow(rays)
plt.show()
plt.imsave('rays.png', rays, cmap='gray')

#%%
theta = np.linspace(0,180, endpoint = False, num=400)
R = operators.Radon(theta)
sinogram =  R(phantom)
plt.imshow(sinogram+np.random.normal(0,0.01,sinogram.shape), cmap='gray')
#plt.axis('auto')
plt.axis('off')
s = sinogram+np.random.normal(0,0.01,sinogram.shape)
plt.imsave('sinogram.png', s, cmap='gray')