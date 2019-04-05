import torch.nn as nn
from functools import reduce


class SKConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 M=2,
                 r=16,
                 L=32):
        super(SKConv,self).__init__()
        d=max(in_channels//r,L)
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels,out_channels,3,stride,padding=1+i,dilation=1+i,groups=32,bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.fc1=nn.Sequential(
            nn.Conv2d(out_channels,d,1,bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True))
        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        #the part of split
        for i,conv in enumerate(self.conv):
            #print(i,conv(input).size())
            output.append(conv(input))
        #the part of fusion
        U=reduce(lambda x,y:x+y,output)
        s=self.global_pool(U)
        z=self.fc1(s)
        a_b=self.fc2(z)
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
        a_b=self.softmax(a_b)
        #the part of selection
        a_b=list(a_b.chunk(self.M,dim=1))#split to a and b
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b))
        V=list(map(lambda x,y:x*y,output,a_b))
        V=reduce(lambda x,y:x+y,V)
        return V


class SKBlock(nn.Module):
    expansion=2
    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None):
        super(SKBlock,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(inplanes,planes,1,1,0,bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True))
        self.conv2=SKConv(planes,planes,stride)
        self.conv3=nn.Sequential(
            nn.Conv2d(planes,planes*self.expansion,1,1,0,bias=False),
            nn.BatchNorm2d(planes*self.expansion))
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self, input):
        shortcut=input
        output=self.conv1(input)
        output=self.conv2(output)
        output=self.conv3(output)
        if self.downsample is not None:
            shortcut=self.downsample(input)
        output+=shortcut
        return self.relu(output)


class SKNet(nn.Module):
    def __init__(self,
                 nums_class=1000,
                 block=SKBlock,
                 nums_block_list=[3, 4, 6, 3]):
        super(SKNet,self).__init__()
        self.inplanes=64
        self.conv=nn.Sequential(nn.Conv2d(3,64,7,2,3,bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(inplace=True))
        self.maxpool=nn.MaxPool2d(3,2,1)
        self.layer1=self._make_layer(block,128,nums_block_list[0],stride=1)
        self.layer2=self._make_layer(block,256,nums_block_list[1],stride=2)
        self.layer3=self._make_layer(block,512,nums_block_list[2],stride=2)
        self.layer4=self._make_layer(block,1024,nums_block_list[3],stride=2)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(1024*block.expansion,nums_class)
        self.softmax=nn.Softmax(-1)

    def forward(self, input):
        output=self.conv(input)
        output=self.maxpool(output)
        output=self.layer1(output)
        output=self.layer2(output)
        output=self.layer3(output)
        output=self.layer4(output)
        output=self.avgpool(output)
        output=output.squeeze(-1).squeeze(-1)
        output=self.fc(output)
        output=self.softmax(output)
        return output

    def _make_layer(self,
                    block,
                    planes,
                    nums_block,
                    stride=1):
        downsample=None
        if stride!=1 or self.inplanes!=planes*block.expansion:
            downsample=nn.Sequential(nn.Conv2d(self.inplanes,planes*block.expansion,1,stride,bias=False),
                                     nn.BatchNorm2d(planes*block.expansion))
        layers=[]
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes=planes*block.expansion
        for _ in range(1,nums_block):
            layers.append(block(self.inplanes,planes))
        return nn.Sequential(*layers)


def SKNet50(nums_class=1000):
    return SKNet(nums_class,SKBlock,[3, 4, 6, 3])


def SKNet101(nums_class=1000):
    return SKNet(nums_class,SKBlock,[3, 4, 23, 3])


# if __name__=='__main__':
#     from PIL import Image
#     from torchvision import transforms
#     from torch.autograd import Variable
#     import torch
#     img=Image.open('timg.jpg').convert('RGB')
#     img=transforms.ToTensor()(img)
#     img=Variable(img).cuda()
#     img=torch.stack([img,img])
#     #img=img.unsqueeze(0)
#     temp=SKNet50().cuda()
#     pred=temp(img)
#     print(pred)

