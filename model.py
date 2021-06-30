#-*-coding:utf-8-*-
# date:2021-6-16
# Author: Mountchicken & YuShu Li
## function: model

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

def reshape(x,type,order=None):
    """
    Args:
        x (tensor): tensor of shape [B,3,100,25,2]
        type (str): 'stacked_rnn','hierachical_rnn','chain','traversal'
        order (list):
            the order to depart joints,
            e.g [[1,8,10,12],[2,9,11,13],...] --> for hierachical_rnn
                [12,10,8,1,3,2,9,11,13,...] --> for chain sequence
                [3,1,8,10,12,10,8,1,2,9,11,13,11,9,2,...] --> for traversal sequence
    Returns:
        x_reshaped: reshaped tensor corresponding with type
    Tricks:
        B -> Batchsize(B)
        A -> Axis(3)
        T -> num of frames(100)
        J -> num of joints(25)
        P -> num of persons(2)
    """
    # Temporal RNN
    if type == 'stacked_rnn':
        #对于stacked_rnn, [B,3,100,25,2]--> [2,B,100,3*25]
        x_reshaped = einops.rearrange(x, 'B A T J P ->P B T (A J)')
    elif type == 'hierarchical_rnn':
        x_reshaped = []
        #对于hierachical_rnn,[B,3,100,25,2] --> 5 * [2,B,100,3*5] 既把25个点按顺序拆分为5份
        for i in range(5):
            x_reshaped.append(einops.rearrange(x[:,:,:,order[i],:],'B A T J P -> P B T (A J)'))

    # Spatial RNN
    elif type == 'chain':
        #对于chain,[B,3,100,25,2] --> [2,B,3*25,100] 25个点的顺序按照chain模式打乱
        x_reshaped = einops.rearrange(x[:,:,:,order,:],'B A T J P -> P B (A J) T')
    elif type == 'traversal':
        #对于traversal, [B,3,100,25,2] --> [2,B,3*47,100] traversal顺序较为特殊，会重复
        x_reshaped = torch.ones((x.shape[0],x.shape[1],x.shape[2],len(order),x.shape[4])).to('cuda') #[B,3,100,47,2]
        for idx, ord in enumerate(order):
            x_reshaped[:,:,:,idx,:] = x[:,:,:,ord,:]
        x_reshaped = einops.rearrange(x_reshaped,'B A T J P -> P B (A J) T ')

    return x_reshaped


class temporal_rnn(nn.Module):

    def __init__(self,order,num_classes,type='hierarchical_rnn'):
        super().__init__()
        self.type = type
        self.order = order
        self.first_rnn_inputsize =[6*3, 6*3, 5*3, 4*3, 4*3]
        assert type in ['stacked_rnn', 'hierarchical_rnn'] # 两种不同的temporal_rnn

        if type == 'stacked_rnn':
            self.rnn1 = nn.LSTM(25*3, 512,batch_first=True)
            self.rnn2 = nn.LSTM(512, 512, batch_first=True)

        elif type == 'hierarchical_rnn':
            self.rnn1=nn.ModuleList()
            for i in range(5):
                self.rnn1.append(
                    nn.LSTM(self.first_rnn_inputsize[i], 128, batch_first=True)
                    )
            # rnn2对应concat后的第二层rnn
            # concat 演着joints维度进行拼接
            self.rnn2 = nn.LSTM(128*5,512,batch_first=True)

        self.fc = nn.Linear(512*100,num_classes)
        nn.init.kaiming_normal_(self.fc.weight)
    def forward(self,x):
        # x shape [B,3,100,25,2]
        # 首先找出一个batch里面哪些是单人动作，哪些是双人动作,默认单人的情况下x[:,:,:,:,1]全为0


        if self.type == 'hierarchical_rnn' :
            # 此情况x_reshaped为列表，其中五个元素对应第一层rnn的五组输入
            x_reshaped = reshape(x,'hierarchical_rnn',self.order) #[B,3,100,25,2]-->5*[2,B,100,3*5]
            x_out1 = [] #第一个人
            x_out2 = [] #第二个人
            #第一次rnn
            for i in range(5):
                x_out1.append(self.rnn1[i](x_reshaped[i][0,:,:,:])[0])
                x_out2.append(self.rnn1[i](x_reshaped[i][1,:,:,:])[0])

            x_out1 = torch.cat(x_out1,dim=2) # [B,100,128] -> [B,100,128*5]
            x_out2 = torch.cat(x_out2,dim=2) # [B,100,128] -> [B,100,128*5]
            #第二层rnn + linear
            #x_out1 = self.fc(self.rnn2(x_out1)[0][:,-1,:]) #[B,100,128*5] -> [B, num_classes]
            x_out1 = self.fc(self.rnn2(x_out1)[0].reshape(x.shape[0],-1))
            #x_out2 = self.fc(self.rnn2(x_out2)[0][:,-1,:]) #[B,100,128*5] -> [B, num_classes]
            x_out2 = self.fc(self.rnn2(x_out2)[0].reshape(x.shape[0],-1))

        elif self.type == 'stacked_rnn' :

            x_reshaped = reshape(x,'stacked_rnn',self.order)  #[B,3,100,25,2]--> [2,B,100,3*25]

            #x_out1 = self.fc(self.rnn2(self.rnn1(x_reshaped[0])[0])[0][:,-1,:])
            x_out1 = self.fc(self.rnn2(self.rnn1(x_reshaped[0])[0])[0].reshape(x.shape[0],-1))
            #x_out2 = self.fc(self.rnn2(self.rnn1(x_reshaped[1])[0])[0][:,-1,:])
            x_out2 = self.fc(self.rnn2(self.rnn1(x_reshaped[1])[0])[0].reshape(x.shape[0],-1))


        return (F.log_softmax(x_out1,1) + F.log_softmax(x_out2,1)) / 2


class spatial_rnn(nn.Module):
    def __init__(self,order,num_classes,type='chain'):
        super().__init__()
        self.type = type
        self.order = order
        assert type in ['chain', 'traversal'] # 两种不同的spatial_rnn
        self.rnn1 = nn.LSTM(100,512,batch_first=True)
        self.rnn2 = nn.LSTM(512,512,batch_first=True)
        self.fc = nn.Linear(512*3*len(order),num_classes)
        nn.init.kaiming_normal_(self.fc.weight)

        
    def forward(self,x):
        # 首先找出一个batch里面哪些是单人动作，哪些是双人动作,默认单人的情况下x[:,:,:,:,1]全为0

        x_reshaped = reshape(x, self.type, self.order)
        #x_out1 = self.fc(self.rnn2(self.rnn1(x_reshaped[0])[0])[0][:,-1,:])
        x_out1 = self.fc(self.rnn2(self.rnn1(x_reshaped[0])[0])[0].reshape(x.shape[0],-1))
        #x_out2 = self.fc(self.rnn2(self.rnn1(x_reshaped[1])[0])[0][:,-1,:])
        x_out2 = self.fc(self.rnn2(self.rnn1(x_reshaped[1])[0])[0].reshape(x.shape[0],-1))

        return (F.log_softmax(x_out1,1) + F.log_softmax(x_out2,1))/2

class two_stream_rnn(nn.Module):
    def __init__(self,temporal_order,temporal_type,spatial_order,spatial_type,num_classes,w=0.9):
        """
        Args:
            temporal_order (list): only used in hierachical_rnn,[[1,8,10,12],[2,9,11,13],...]
            temporal_type (str): 'hierarchical_rnn' or 'stacked_rnn'
            spatial_order (list): [12,10,8,1,3,2,9,11,13,...]
            spatial_type (str): 'chain' or 'traversal'
            num_classes (int): num of prediction
        """
        super().__init__()
        self.spatial_rnn = spatial_rnn(spatial_order,num_classes,spatial_type)
        self.temporal_rnn = temporal_rnn(temporal_order,num_classes,temporal_type)
        self.w = w

    def forward(self,x):

        spatial_score = self.spatial_rnn(x)
        temporal_score = self.temporal_rnn(x)
        total_score = temporal_score * self.w + spatial_score* (1 - self.w)
        

        return total_score
 

if __name__ == "__main__":
    x = torch.rand(10,3,100,25,2).to('cuda')
    label = torch.Tensor([1,2,3,4,5,6,7,8,9,10]).to('cuda')
    criterion = nn.NLLLoss()
    """temporal rnn"""
    temporal_type = 'hierarchical_rnn'
    #temporal_type = 'stacked_rnn'
    stacked_rnn_order = None
    hierachical_rnn_order = [[8,9,10,11,23,24], #左臂 6点
                      [4,5,6,7,21,22], # 右臂 6点
                      [3,2,20,1,0], #躯干 5点
                      [16,17,18,19],#左腿 4点
                      [12,13,14,15], #右腿 4点
    ]
    """spatial rnn"""
    spatial_type = 'traversal'
    #spatial_type = 'chain'
    traversal_rnn_order = [20,8,9,10,11,23,24,23,11,10,9,8,4,5,6,7,21,22,21,7,6,5,4,20,
                           2,3,2,1,0,
                           16,17,18,19,18,17,16,0,12,13,14,15,14,13,12,0,
                           1,12
                            ] # 47 joints
    chain_rnn_order = [23,24,11,10,9,8,20,4,5,6,7,21,22, #①
                        3,2,20,1,0, #②
                        19,18,17,16,0,12,13,14,15 ] #③
    num_classes = 60
    model = two_stream_rnn(
                            temporal_order = hierachical_rnn_order,
                            temporal_type = temporal_type,
                            spatial_order = traversal_rnn_order,
                            spatial_type = spatial_type,
                            num_classes = num_classes
                            ).to('cuda')
    pred = model(x)
    loss = criterion(pred,label.long())
    loss.backward()
    print(pred.shape)



