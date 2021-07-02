import torch
import torch.nn.functional as F 
import torch.nn as nn
from feeder import Feeder
from model import two_stream_rnn
from tqdm import tqdm
device='cuda' if torch.cuda.is_available() else 'cpu'

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

if __name__=='__main__':
    #load testset
    print('Load Testset')
    data_path = "NTU-RGB-D/xview/val_data.npy"
    label_path = "NTU-RGB-D/xview/val_label.pkl"
    num_frame_path = "NTU-RGB-D/xview/val_num_frame.npy"
    test_set = Feeder(data_path, label_path, num_frame_path, random_valid_choose=False,
                        random_shift=False,
                        random_move=False,
                        window_size=100,
                        normalization=False,
                        debug=False,
                        origin_transfer=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=512,
        drop_last=False,
        shuffle=False,
        num_workers=1, 
        pin_memory=True)
    print('data loaded')
    # Initial network
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
    #load checkpoint
    ckp = torch.load('two_stream_with_transform_crossview.pt',map_location='cuda')
    #ckp = torch.load('two_stream_no_transform_crossview.pt',map_location='cuda')
    network.load_state_dict(ckp['network'])
    criterion = nn.NLLLoss()
    total_loss, total_correct = test(network,test_loader,criterion)
    print('Test acc: ',total_correct/len(test_set))
    print('Test Loss: ',total_loss/len(test_loader))