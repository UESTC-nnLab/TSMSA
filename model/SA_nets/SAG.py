import torch
import torch.nn as nn

from .BaseConv import BaseConv
# from BaseConv import BaseConv
import time


class mlp(nn.Module):

    def __init__(self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = BaseConv(in_channels, hidden_channels, 1, 1)
        self.fc2 = BaseConv(hidden_channels, out_channels, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SAG_atten(nn.Module):
    def __init__(self, dim, bias=False, proj_drop=0.):
        super().__init__()

        self.fc1 = BaseConv(dim, dim, 1, 1, bias=bias)  
        
        self.fc2 = BaseConv(dim, dim, 3, 1, bias=bias)  
        
        self.mix = BaseConv(2*dim, dim, 3, 1, bias=bias) 
        self.reweight = mlp(dim, dim, dim)
        

    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_1 = x.clone()
        
        # 负数为零
        x_1[x_1<0] = 0
        for i in range(B):
            for j in range(C):
                mean = x_1[i,j,:,:].mean() 
                x_1[i,j,:,:] = x[i,j,:,:]/(mean + 1e-4)   

        x_1 = self.fc1(x_1)
        
        x_2 = self.fc2(x)
        
        x_1 = self.mix(torch.cat([x_1, x_2], dim=1))
        x_1 = self.reweight(x_1)
        
        x = residual * x_1
        return x
    
class SAG_atten_up(nn.Module):
    def __init__(self, dim, bias=False):
        super().__init__()
        self.fc1 = BaseConv(dim, dim, 1, 1, bias=bias)  
        self.fc2 = BaseConv(dim, dim, 1, 1, bias=bias)  
        self.fc3 = BaseConv(dim, dim, 1, 1, bias=bias)
        self.fc = BaseConv(dim, dim, 3, 1, bias=bias)
        
        self.mix = BaseConv(4*dim, dim, 3, 1, bias=bias) 
        self.reweight = mlp(dim, dim, dim)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        x_1 = torch.max(x,torch.tensor([0.]))
        x = torch.cat([x] + [self.SagMean(H,W,x_1,size) for size in [1,2,4]], dim=1)
        x = self.mix(x)
        x = self.reweight(x)
        return x*residual
    
    def SagMean(self, H, W, x, size=1):
        h, w = H//size, W//size
        x_ = x.clone()
        for i in range(size):
            for j in range(size):
                mean = x_[:, :, i*h:(i+1)*h, j*w:(j+1)*w].mean()
                x_[:, :, i*h:(i+1)*h, j*w:(j+1)*w] = x_[:, :, i*h:(i+1)*h, j*w:(j+1)*w]/(mean+1e-4) 
        if size == 1:
            x = self.fc1(x_)
        elif size == 2:
            x = self.fc2(x_)
        else:
            x = self.fc3(x_)                
        return x

if __name__ == "__main__":
    
    x = torch.rand([2,32,640,640])
    t = time.time()
    x = SAG_atten_up(32)(x)
    t = time.time() - t
    print(x.shape, t)