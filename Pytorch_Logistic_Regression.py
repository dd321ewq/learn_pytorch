import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=torch.nn.Linear(1,1)
    def forward(self,x):
        y_pred=F.sigmoid(self.linear(x))
        return y_pred



model=LogisticRegressionModel()

criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.SGD(model.parameters(),lr=0.05)

epoch_list=[]
cost_list=[]

for epoch in range(10000):
    y_pred=model(x_data)
    loss=criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_list.append(epoch)
    cost_list.append(loss.item())  # loss


plt.plot(epoch_list,cost_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()


# x_test=torch.Tensor([[2.5]])
# print('predict(after training)','x=4 y=',model(x_test).item())  ###训练后对x=4.0的y值预测


x=np.linspace(0,10,200)  #0~10取200个点
x_test=torch.Tensor(x).view((200,1))  #200行1列
y_test=model(x_test)
y=y_test.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')  #0~10处 0.5~0.5划红线
plt.xlabel('Hours')
plt.ylabel("Probability of Pass")
plt.grid()
plt.show()







if __name__ == "__main__":
    main()