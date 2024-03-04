import matplotlib.pyplot as plt
import torch

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#随机梯度下降拟合y=wx

w = torch.Tensor([1.0])  #初始权值为1,设置为Tensor类型
w.requires_grad=True #设置w需要反向传播

a=0.01 #学习率为0.01

def forward(x):     #求当前权值下预测值
    return x*w   #运算符重载


def loss(x,y):    ##求单个点的loss值
    y_pred=forward(x)
    return (y_pred-y)**2

# def gradient(x,y):
#     grad=2*x*(x*w-y)
#     return grad


epoch_list=[]
cost_list=[]
print('predict(before training)','x=4 y=',forward(4).item())  ###训练前对x=4.0的y值预测

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        loss_val=loss(x,y)  #一个点求loss
        loss_val.backward() #自动反向传播，自动求计算图需要求梯度的权值，并存到w.grad中，并释放计算图
       # grad_val=gradient(x,y)  #根据已推导出的公式求梯度
        print('\tgrad:',x,y,w.grad.item())
        w.data=w.data-a*w.grad.data  #更新权值,要加data，不然又创建计算图了
        w.grad.data.zero_() #梯度清零

    print('epoch:',epoch,'w=',w.item(),'loss=',loss_val.item())
    epoch_list.append(epoch)
    cost_list.append(loss_val.item()) #一轮中最后一个样本的loss


print('predict(after training)','x=4 y=',forward(4).item())  ###训练后对x=4.0的y值预测

plt.plot(epoch_list,cost_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


if __name__ == "__main__":
    main()
