import torch


if __name__=='__main__':
    batch_size = 1  #批次
    seq_len = 3   #有三天的数据
    input_size=4  #每天天气的特征数
    hidden_size =2 #隐藏层维度
    num_layers=1

    cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

    inputs=torch.randn(seq_len,batch_size,input_size)
    hidden = torch.zeros(num_layers,batch_size,hidden_size) #第一个隐藏层是0向量

    out,hidden = cell(inputs,hidden)  #不用写循环了

    print('Output size:',out.shape)
    print('Output:',out)
    print('Hidden size',hidden.shape)
    print('Hidden:',hidden)