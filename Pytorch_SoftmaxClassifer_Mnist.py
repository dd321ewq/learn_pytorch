import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import transforms,datasets

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
                        batch_size=batch_size,
                        drop_last=False
                        )

test_dataset=datasets.MNIST(root='./mnist',
                             train=False,
                             download=True,
                             transform=transform
                             )

test_loader = DataLoader(test_dataset,
                        shuffle=False,
                        batch_size=batch_size,
                        drop_last=False
                         )


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1=torch.nn.Linear(784,512)
        self.l2=torch.nn.Linear(512,256)
        self.l3=torch.nn.Linear(256,128)
        self.l4=torch.nn.Linear(128,64)
        self.l5=torch.nn.Linear(64,10)

    def forward(self,x):
        #最后一个是torch.Size([16, 784]) 其他是64*784   因为最后一个批次只有16个
        x=x.view(-1,784)         #将tensor展平   -1代表自动计算行的维度   64*784
        #x=x.view(batch_size,-1)    #也是64*784为什么这个不行   #原因：最后一个batch的数据少于一个batchsize 解决方式：设置dataloader中drop_last=True
        #print(x.shape)
        x=F.relu_(self.l1(x))
        x=F.relu_(self.l2(x))
        x=F.relu_(self.l3(x))
        x=F.relu_(self.l4(x))

        return self.l5(x)   #最后一层不激活


model=Net()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)  #动量0.5



epoch_list=[]
cost_list=[]
count=0.0
def train(epoch):
    running_loss = 0.0
    for batch_index,data in enumerate(train_loader,0):
        inputs,target=data
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
    for epoch in range(10):
        train(epoch)
        test()
    plt.plot(epoch_list,cost_list)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()









