import torch
import matplotlib.pyplot as plt
import numpy as np



xy=np.loadtxt('./diabetes.csv',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])      #最后一列不要 ，向量形式
y_data = torch.from_numpy(xy[:,[-1]])       #只要最后一列，且为矩阵形式


class Multiple_DimensionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,1)
        self.sigmoid=torch.nn.Sigmoid()  # 将其看作是网络的一层，而不是简单的函数使用

    def forward(self,x):
        x=self.sigmoid(self.linear1(x))
        x=self.sigmoid(self.linear2(x))
        x=self.sigmoid(self.linear3(x))
        return x



model=Multiple_DimensionModel()

criterion=torch.nn.BCELoss(size_average=True)
optimizer=torch.optim.Adam(model.parameters(),lr=0.05)

epoch_list=[]
cost_list=[]

for epoch in range(5000):
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



if __name__ == "__main__":
    main()