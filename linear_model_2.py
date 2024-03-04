import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#这里设函数为y=wx+b
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):
    return x * w + b

def loss(x,y):
    y_pred = forward(x)
    return (y_pred-y)*(y_pred-y)

mse_list = []
W=np.arange(0.0,4.1,0.1)
B=np.arange(0.0,4.1,0.1)
[w,b]=np.meshgrid(W,B)    #w和b都是41*41的矩阵，不过w是从W的角度，b是从B的角度

l_sum = 0
for x_val, y_val in zip(x_data, y_data):     #用了meshgrid只需要循环3次
    y_pred_val = forward(x_val)    #这样可以一次计算41*41个
    print(y_pred_val)
    loss_val = loss(x_val, y_val)    #一次计算41*41个loss
    l_sum += loss_val               #总共加3次，3个点的loss矩阵加起来

fig = plt.figure()   #创建画板
ax = Axes3D(fig)   #3d图形
ax.plot_surface(w, b, l_sum/3)
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('mse')

plt.show()



if __name__ == "__main__":
    main()
