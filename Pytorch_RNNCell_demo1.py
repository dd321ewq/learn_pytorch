import torch


if __name__=='__main__':
    batch_size = 1  #批次
    seq_len = 3   #有三天的数据
    input_size=4  #每天天气的特征数
    hidden_size =2 #隐藏层维度

    cell = torch.nn.RNNCell(input_size=input_size,hidden_size=hidden_size)

    dataset=torch.randn(seq_len,batch_size,input_size)
    hidden = torch.zeros(batch_size,hidden_size) #第一个隐藏层是0向量

    for idx,input in enumerate(dataset):
        print('='*20,idx,'='*20)
        print('input size:',input.shape)

        hidden=cell(input,hidden)  #rnn自动更新隐藏层

        print('outputs size:',hidden.shape)
       # print(hidden)

