import torch.nn as nn
import torchvision.models as models
import torch
from torch.nn import functional as F
# import fvcore.nn.weight_init as weight_init


# class CLModule(nn.Module):
        
#     def __init__(self):
#         super(CLModule, self).__init__()
#         self.Linear1 = nn.Linear(512*5*5,512)
#         self.Linear2 = nn.Linear(512,256)

#     def forward(self, x, batch):

#         for num in range(batch):
#             if num == 0:
#                 out = F.adaptive_max_pool2d(x[num],(5,5)).unsqueeze(0)
#             else:
#                 outTem = F.adaptive_max_pool2d(x[num],(5,5)).unsqueeze(0)
#                 out = torch.cat((out,outTem),0)

#         out = out.view(batch,-1)
#         out = self.Linear1(out)
#         out = self.Linear2(F.relu(out))
#         return out

# if __name__ == '__main__':
#     CLNet = CLModule()
#     x = torch.rand(4,512,5,5)
#     out = CLNet(x)
#     print(out.shape)

class CLModule(nn.Module):
        
    def __init__(self):
        super(CLModule, self).__init__()
        self.Linear1 = nn.Linear(512*5*5,512)
        self.Linear2 = nn.Linear(512,256)
        self.head = nn.Sequential(
            nn.Linear(512*5*5,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
        )
        for layer in self.head:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=1)
                # weight_init.c2_xavier_fill(layer)

    def forward(self, x, batch):

        for num in range(batch):
            if num == 0:
                out = F.adaptive_max_pool2d(x[num],(5,5)).unsqueeze(0)
            else:
                outTem = F.adaptive_max_pool2d(x[num],(5,5)).unsqueeze(0)
                out = torch.cat((out,outTem),0)
        out = out.view(batch,-1)
        feat = self.head(out)
        feat_normalized = F.normalize(feat, dim=1)
        return feat_normalized

if __name__ == '__main__':
    CLNet = CLModule()
    x = torch.rand(4,512,5,5)
    out = CLNet(x)
    print(out.shape)