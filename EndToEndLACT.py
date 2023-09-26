import torch
import torch.nn as nn
from torch.nn.functional import relu
import numpy as np
import skimage as ski
from skimage.draw import disk
import matplotlib.pyplot as plt


class Square:
    def __init__(self, img_size=100, num_imgs=1):
        self.img_size = [img_size, img_size]

    def __call__(self,):
        S = np.zeros(self.img_size)
        x_start, y_start = np.random.randint(self.img_size[1]//3,self.img_size[1]//2 , size=(2,))
        x_end, y_end = np.random.randint(self.img_size[1]//2, 2*self.img_size[1]//3, size=(2,))
        S[x_start:x_end, y_start:y_end] = 1
        return S
    
class Circle:
    def __init__(self, img_size=100):
        self.img_size = [img_size, img_size]
        
    def __call__(self,):
        C = np.zeros(self.img_size)
        radius = np.random.randint(self.img_size[1]//5, self.img_size[1]//4)
        row = self.img_size[1]//2 + np.random.randint(-self.img_size[1]//4, self.img_size[1]//4)
        col = self.img_size[1]//2 + np.random.randint(-self.img_size[1]//4, self.img_size[1]//4)
        # modern scikit uses a tuple for center
        rr, cc = disk((row, col), radius)
        C[rr, cc] = 1.
        return C

#%%    


class Recon(nn.Module):
    def __init__(self, img_size=100, thetas=None):
        super(Recon, self).__init__()
        self.img_size = img_size
        self.thetas = thetas if not thetas is None else np.linspace(0,180, endpoint = False, num=100)
        self.input_size = [img_size, len(self.thetas)]


        #self.linear = nn.Linear(img_size * len(self.thetas), img_size**2)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64,32, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(32,16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(16,1 , kernel_size=1, stride=1, padding=0)
        self.activation = nn.ReLU()
        
        
    def __call__(self, S):
        if S.ndim == 2:
            S = S[None,...]

        S = torch.Tensor(S)
        #S = S.reshape(-1, img_size * len(self.thetas))
        #S = self.linear(S)
        #S = self.activation(S)
        S = S.reshape(-1, 1, self.img_size, self.img_size)
        S = self.conv1(S)
        S = self.activation(S)
        S = self.conv2(S)
        S = self.activation(S)
        S = self.conv3(S)
        S = self.activation(S)
        S = self.conv4(S)
        S = self.activation(S)
        S = self.conv5(S)
        S = self.activation(S)
        S = self.conv6(S)
        S = self.activation(S)# + S1
        return S
   
    

        
class UNet(nn.Module):
    
    def __init__(self, n_class=1):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # output: 570x570x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding=1) # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # output: 282x282x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # output: 280x280x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # output: 138x138x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding=1) # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 68x68x256

        # # input: 68x68x256
        # self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 66x66x512
        # self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 64x64x512
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(256, 512, kernel_size=3, padding=1) # output: 30x30x1024
        self.e52 = nn.Conv2d(512, 512, kernel_size=3, padding=1) # output: 28x28x1024


        # # Decoder
        # self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        # self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x[None,  None,...]
        if x.ndim == 3:
            x = x[:, None,...]

        x = torch.Tensor(x)
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        # xe41 = relu(self.e41(xp3))
        # xe42 = relu(self.e42(xe41))
        # xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp3))
        xe52 = relu(self.e52(xe51))
        
        # # Decoder
        # xu1 = self.upconv1(xe52)
        # xu11 = torch.cat([xu1, xe42], dim=1)
        # xd11 = relu(self.d11(xu11))
        # xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xe52)#d12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    
#%% Training
img_size = 64
num_thetas = 10
theta = np.linspace(0,90, endpoint = False, num=num_thetas)
net = Recon(img_size=img_size, thetas=theta)
net = UNet()

loss_fct = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
batch_size = 5

train_shape = Circle(img_size=img_size)
#SQ = Square()
 
for i in range(1000):
    optimizer.zero_grad()
   
    inp = np.zeros((batch_size, 1, img_size, img_size))
    I = np.zeros((batch_size, 1, img_size, img_size))
    
    for i in range(batch_size):
        I[i,0,...] = train_shape()
        sinogram =  ski.transform.radon(I[i,0,...], theta)
        inp[i, 0, ...] = ski.transform.iradon(sinogram, theta=theta)
    Irecon = net(inp)
    loss = loss_fct(Irecon, torch.Tensor(I))
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    loss1 = loss.item()
    loss2 = loss_fct(torch.Tensor(inp), torch.Tensor(I)).item()
    print(12*'-')
    print('Current Loss:' + str(loss1))
    print('IRadon Loss:' + str(loss2))
    for param_group in optimizer.param_groups:
        print('Current lr:' + str(param_group['lr']))
    
    
# %% Test
img_size=64
SQ = Square(img_size=img_size)

C = Circle(img_size=img_size)
CI = C()
SQI = SQ()

CS =  ski.transform.radon(CI, theta)
SQS =  ski.transform.radon(SQI, theta)


CIrad = ski.transform.iradon(CS, theta=theta)
CIrecon = net(CIrad)
SQIrad = ski.transform.iradon(SQS, theta=theta)
SQIrecon = net(SQIrad)

#%%
plt.close('all')

fig, ax = plt.subplots(2,4)
ax[0,0].imshow(SQI)
ax[0,1].imshow(SQS)
ax[0,2].imshow(SQIrecon.detach()[0,0,...])
ax[0,3].imshow(SQIrad)

ax[1,0].imshow(CI)
ax[1,1].imshow(CS)
ax[1,2].imshow(CIrecon.detach()[0,0,...])
ax[1,3].imshow(CIrad)