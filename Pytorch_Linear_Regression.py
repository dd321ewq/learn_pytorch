import matplotlib.pyplot as plt
import torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

#用pytorch框架随机梯度下降拟合y=wx

class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(1,1)
        # (1,1)是指输入x和输出y的特征维度，这里数据集中的x和y的特征都是1维的
        # 该线性层需要学习的参数是w和b  获取w/b的方式分别是~linear.weight/linear.bias

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

model=LinearModel()

criterion = torch.nn.MSELoss(size_average=True) #求loss
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)  #学习率为0.01 随机梯度下降


epoch_list=[]
cost_list=[]

x_test=torch.Tensor([[4.0]])
print('predict(before training)','x=4 y=',model(x_test).item())  ###训练前对x=4.0的y值预测

for epoch in range(100):
        y_pred=model(x_data)
        loss_val=criterion(y_pred,y_data)  #求MSEloss
        print('epoch:', epoch,  'loss=', loss_val.item())

        loss_val.backward()                #反向传播
        optimizer.step()        #更新权值
        optimizer.zero_grad()   #清零

        epoch_list.append(epoch)
        cost_list.append(loss_val.item())  # loss

print('w=',model.linear.weight.item())  #w
print('b=',model.linear.bias.item())    #b


print('predict(after training)','x=4 y=',model(x_test).item())  ###训练后对x=4.0的y值预测

plt.plot(epoch_list,cost_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


if __name__ == "__main__":
    main()
