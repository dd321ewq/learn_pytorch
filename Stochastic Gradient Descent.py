import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#随机梯度下降拟合y=wx

w = 1  #初始权值为1
a=0.01 #学习率为0.01

def forward(x):     #求当前权值下预测值
    return x*w


def loss(x,y):    ##求单个点的loss值
    y_pred=forward(x)
    return (y_pred-y)**2

def gradient(x,y):
    grad=2*x*(x*w-y)
    return grad


epoch_list=[]
cost_list=[]
print('predict(before training)','x=4 y=',forward(4))  ###训练前对x=4.0的y值预测

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        loss_val=loss(x,y)  #一个点求loss
        grad_val=gradient(x,y)  #根据已推导出的公式求梯度
        w=w-a*grad_val  #更新权值


    print('epoch:',epoch,'w=',w,'loss=',loss_val)
    epoch_list.append(epoch)
    cost_list.append(loss_val) #一轮中最后一个样本的loss


print('predict(after training)','x=4 y=',forward(4))  ###训练后对x=4.0的y值预测

plt.plot(epoch_list,cost_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


if __name__ == "__main__":
    main()
