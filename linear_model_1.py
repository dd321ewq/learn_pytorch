import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

####实现线性模型y=wx拟合上述数据并画图


def loss(x, y,y_pred):
    return (y_pred - y) ** 2   #某点的loss是差值的平方

def main():
    # 穷举法
    w_list = []
    mse_list = []
    for w in np.arange(0.0, 4.1, 0.1):   #w从0.0到4.0步长0.1
        print("w=", w)
        loss_sum = 0
        for x_val, y_val in zip(x_data, y_data):   #zip函数
            y_pred_val = x_val*w  #在当前权值下计算预测值
            loss_val = loss(x_val, y_val,y_pred_val) #计算当前点的loss值
            loss_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', loss_sum / 3)   #计算平均loss值

        w_list.append(w)
        mse_list.append(loss_sum / 3)

    plt.plot(w_list, mse_list)
    plt.ylabel('Loss')
    plt.xlabel('w')
    plt.show()


if __name__ == "__main__":
    main()
