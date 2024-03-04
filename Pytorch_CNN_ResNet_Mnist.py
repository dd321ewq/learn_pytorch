import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms,datasets

device = torch.device("cuda:0"if torch.cuda.is_available()else "cpu")




batch_size = 64
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
])    #像素0~255转变成0~1tensor并正态化   两个数分别是均值和方差

train_dataset=datasets.MNIST(root='./mnist',
                             train=True,
                             download=True,
                             transform=transform
                             )

train_loader = DataLoader(train_dataset,
                        shuffle=True,
                        batch_size=batch_size
                        )

test_dataset=datasets.MNIST(root='./mnist',
                             train=False,
                             download=True,
                             transform=transform
                             )

test_loader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size
                        )



class ResidualBlock(torch.nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels,in_channels,kernel_size=3,padding=1)  #输入通道和输出通道数要一致

    def forward(self,x):

        y=F.relu_(self.conv1(x))
        y=self.conv2(y)

        return  F.relu_(x+y)

class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)   #池化层

        self.resblock1=ResidualBlock(in_channels=16)
        self.resblock2=ResidualBlock(in_channels=32)

        self.fc = torch.nn.Linear(512,10)


    def forward(self,x):

        batch_size=x.size(0)

        x=self.pooling(F.relu_(self.conv1(x)))
        x=self.resblock1(x)
        x=self.pooling(F.relu_(self.conv2(x)))
        x=self.resblock2(x)

        x=x.view(batch_size,-1)
        #print(x.shape)
        x=self.fc(x)
        return x  #最后一层不激活


model=Net()
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)  #动量0.5



epoch_list=[]
cost_list=[]
count=0.0
def train(epoch):
    running_loss = 0.0
    for batch_index,data in enumerate(train_loader,0):
        inputs,target=data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()


        running_loss+=loss.item()
        count=batch_index


    epoch_list.append(epoch)
    cost_list.append(running_loss/count)


def test():
    correct = 0
    total = 0
    with torch.no_grad():           #测试不需要计算梯度
        for data in test_loader:
            images,labels=data  #一次验证64个
            outputs = model(images)   #outputs是一个矩阵64*10   即batchsize*10
            _,predicted=torch.max(outputs.data,dim=1)  #找每一行最大值和对应的下标，这里只要下标矩阵，最大值的下标就是预测的数
            total+=labels.size(0)   #label是矩阵[N,1]
            correct +=(predicted==labels).sum().item()   #猜对的数目
        print('Accuracy on test set:%d %%'%(100*correct/total))


if __name__=='__main__':
    print(torch.cuda.is_available())

    # from opacus import PrivacyEngine
    #
    # privacy_engine = PrivacyEngine()
    # model, optimizer, train_loader = privacy_engine.make_private(
    #     module=model,
    #     optimizer=optimizer,
    #     data_loader=train_loader,
    #     noise_multiplier=1.1,
    #     max_grad_norm=1.0,
    # )

    for epoch in range(10):
        train(epoch)
        test()
    plt.plot(epoch_list,cost_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()

