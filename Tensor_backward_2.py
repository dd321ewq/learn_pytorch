import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#随机梯度下降拟合y=w1x^2+w2x+b

w1 = torch.Tensor([1.0])  #初始权值为1,设置为Tensor类型
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])

w1.requires_grad=True #设置w1需要反向传播更新
w2.requires_grad=True #设置w2需要反向传播更新
b.requires_grad=True #设置b需要反向传播更新

a=0.001 #学习率为0.01

def forward(x):     #求当前权值下预测值
    return x**2*w1+x*w2+b                  #运算符重载

def loss(x,y):    ##求单个点的loss值
    y_pred=forward(x)
    return (y_pred-y)**2

epoch_list=[]
cost_list=[]
print('predict(before training)','x=4 y=',forward(4).item())  ###训练前对x=4.0的y值预测

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        loss_val=loss(x,y)  #一个点求loss
        loss_val.backward() #自动反向传播，自动求计算图需要求梯度的权值，并存到w.grad中，并释放计算图
        print('\tgrad:',x,y,w1.grad.item(),w2.grad.item(),b.grad.item())
        w1.data=w1.data-a*w1.grad.item()  #更新权值,要加data，不然又创建计算图了
        w2.data=w2.data-a*w2.grad.item()
        b.data=b.data-a*b.grad.item()
        w1.grad.data.zero_() #梯度清零
        w2.grad.data.zero_()
        b.grad.data.zero_()

    print('epoch:',epoch,'loss=',loss_val.item())
    epoch_list.append(epoch)
    cost_list.append(loss_val.item()) #一轮中最后一个样本的loss


print('predict(after training)','x=4 y=',forward(4).item())  ###训练后对x=4.0的y值预测

plt.plot(epoch_list,cost_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


if __name__ == "__main__":
    main()
