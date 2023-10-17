import matplotlib.pyplot as plt
import numpy as np
import imageio

#%%
plt.close('all')
fig, ax = plt.subplots()
frames = []
for i in range(49): 
    I = plt.imread(str(i+1) + '.png')
    frames.append(I.copy())
    
    print(i)
    #plt.pause(0.1)
    
#%%
imageio.mimsave('./ct.gif', # output gif
                frames,          # array of input frames
                fps = 1)  

