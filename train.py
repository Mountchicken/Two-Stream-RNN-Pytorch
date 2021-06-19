import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from feeder import Feeder
from model import two_stream_rnn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(network,testloader,criterion):
    print('evaluating')
    network.eval()
    total_correct=0
    total_loss=0
    with torch.no_grad():
        for images,labels in tqdm(testloader):
            images = images.to(device).float()
            labels = labels.to(device)
            preds = network(images)
            loss = criterion(preds,labels)
            total_loss += loss.item()
            total_correct += preds.argmax(dim=1).eq(labels).sum().item()
    network.train()
    return total_loss,total_correct

'''采用边训练，边测试的流程，保存测试集最高准去率模型'''

def train():
    # Hyperparameters:
    batch_size = 256
    learning_rate = 0.01
    num_epochs = 80
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load Trainset
    print('Load Trainset')
    data_path = "NTU-RGB-D/xview/val_data.npy"
    label_path = "NTU-RGB-D/xview/val_label.pkl"
    num_frame_path = "NTU-RGB-D/xview/val_num_frame.npy"
    train_set = Feeder(data_path, label_path, num_frame_path, random_valid_choose=False,
                     random_shift=False,
                     random_move=False,
                     window_size=100,
                     normalization=True,
                     debug=False,
                     origin_transfer=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1, 
        pin_memory=True)

    # load Testset
    print('Load Testset')
    data_path = "NTU-RGB-D/xview/val_data.npy"
    label_path = "NTU-RGB-D/xview/val_label.pkl"
    num_frame_path = "NTU-RGB-D/xview/val_num_frame.npy"
    test_set = Feeder(data_path, label_path, num_frame_path, random_valid_choose=False,
                     random_shift=False,
                     random_move=False,
                     window_size=100,
                     normalization=True,
                     debug=False,
                     origin_transfer=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1, 
        pin_memory=True)
    print('data loaded')

    #Model Definition
    print('Initializing Network')

    """temporal rnn"""
    temporal_type = 'hierarchical_rnn'
    #temporal_type = 'stacked_rnn'
    stacked_rnn_order = None
    hierachical_rnn_order = [[8,9,19,11,23,24], #左臂 6点
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
                        19,18,17,16,12,13,14,15 ] #③
    num_classes = 60
    network = two_stream_rnn(
                            temporal_order = hierachical_rnn_order,
                            temporal_type = temporal_type,
                            spatial_order = traversal_rnn_order,
                            spatial_type = spatial_type,
                            num_classes = num_classes
                            )
    network = network.to(device)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()
    comment=f'TwoStreamRNN batch_size={batch_size} lr={learning_rate} device={device}'
    tb=SummaryWriter(comment=comment)

    best_acc = 0.0
    for epoch in range(num_epochs):
        train_loss = 0
        train_correct = 0
        loop = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        for _, (images, labels) in loop:
            images = images.to(device).float()
            labels = labels.to(device)
            optimizer.zero_grad()
            preds = network(images)
            loss = criterion(preds, labels)
            loss.backward()
            for group in optimizer.param_groups:
                for param in group['params']:
                    param.grad.data.clamp_(-5,5)
            optimizer.step()
            train_loss += loss.item()
            train_correct += preds.argmax(dim=1).eq(labels).sum().item()
            loop.set_description(f"Epoch[{epoch}/{num_epochs}]")
        trainset_acc = train_correct / len(train_set)
        train_avg_loss = train_loss / len(train_loader)
        test_loss , test_correct = test(network,test_loader,criterion)
        testset_acc = test_correct / len(test_set)
        test_avg_loss = test_loss / len(test_loader)

        tb.add_scalar('Train Loss',train_avg_loss,epoch)
        tb.add_scalar('Accuracy on Trainset',trainset_acc,epoch)
        tb.add_scalar('Test Loss',test_avg_loss,epoch)
        tb.add_scalar('Accuracy on Testset',testset_acc,epoch)

        '''只保存测试集准确率不断上升的epoch'''
        if testset_acc > best_acc:
            torch.save(network.state_dict(),'./best_model.pt')
            best_acc = testset_acc
        print("epoch: ",epoch,"loss: ",train_avg_loss,"Train acc: ",trainset_acc,"Test acc: ",testset_acc)
    tb.close()

if __name__=='__main__':
    train()