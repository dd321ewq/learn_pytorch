import matplotlib.pyplot as plt


x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#梯度下降拟合y=wx

w = 1  #初始权值为1
a=0.01 #学习率为0.01

def forward(x):     #求当前权值下预测值
    return x*w


def cost(xs,ys):    ##求mse
    cost=0
    for x,y in zip(xs,ys):
        y_pred=forward(x)
        cost+=(y_pred-y)**2
    return cost/len(xs)

def gradient(xs,ys):
    grad=0
    for x,y in zip(xs,ys):
        grad+=2*x*(x*w-y)
    return grad/len(xs)


epoch_list=[]
cost_list=[]
print('predict(before training)','x=4 y=',forward(4))  ###训练前对x=4.0的y值预测

for epoch in range(100):
    cost_val=cost(x_data,y_data)  #求出平均损失
    grad_val=gradient(x_data,y_data)  #根据已推导出的公式求梯度
    w=w-a*grad_val  #更新权值
    print('epoch:',epoch,'w=',w,'loss=',cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)


print('predict(after training)','x=4 y=',forward(4))  ###训练后对x=4.0的y值预测

plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()


if __name__ == "__main__":
    main()
