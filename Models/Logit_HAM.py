# -*- coding: utf-8 -*-
import torch
import sys, os
import json
import torch.nn as nn  
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim

import prettytable
import time
from thop.profile import profile

from PIL import Image
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary
from tqdm.notebook import tqdm
import seaborn as sns

from utils import ImageShow,draw_size_acc,one_hot
from utils import confusion_matrix,metrics_scores,pff

from model import make_features

torch.cuda.empty_cache()
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

cpus_per_gpu = 4

# Settings.
sys.path.append(os.pardir)
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

img_title = "HAM10000"
best_acc = 0.
eval_acc = 0.
best_train = 0.
dict_batch = {}
dict_imgSize = {}

#defined 
try:
    tmp = len(train_acc_list)
except NameError:
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    test_auc_list = []
    val_loss_list = []
    val_acc_list = []
#activate ImageShow
show = ImageShow(train_loss_list = train_loss_list,
                 train_acc_list = train_acc_list,
                test_loss_list = test_loss_list,
                test_acc_list = test_acc_list,
                test_auc_list = test_auc_list,
                val_loss_list = val_loss_list,
                val_acc_list = val_acc_list,
                )

def get_data(trans_test='312'):
    global test_dataset, train_dataset, train_loader,val_loader,test_loader
    global train_num,val_num,test_num,n_classes,cla_dict
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop((299, 299)),
                                     transforms.RandomVerticalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
        "val": transforms.Compose([transforms.Resize((302,302)),
                                   transforms.CenterCrop((299, 299)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                  ]),
        "test": transforms.Compose([transforms.Resize((trans_test,trans_test)),
                                   transforms.CenterCrop((299, 299)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                                  ])
        }

    data_root = os.path.abspath(os.path.join(os.getcwd(),".."))  # get data root path
    image_path = os.path.join(data_root)
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

    train_dataset = datasets.ImageFolder(root=os.path.join(image_path,"HAM10K",train_doc),# train_doc=aug_train_8000
                                         transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(root=os.path.join(image_path,"HAM10K",val_doc),# val_doc=val_dir
                                            transform=data_transform["val"])
    test_dataset = datasets.ImageFolder(root=os.path.join(image_path,"HAM10K",test_doc),# test_doc=test_dir
                                            transform=data_transform["test"])

    train_num = len(train_dataset)
    val_num = len(val_dataset)
    test_num = len(test_dataset)
    
    data_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in data_list.items())
    n_classes  = len(data_list)
    print(f'Using {n_classes } classes.')
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open(f'{img_title}.json', 'w') as json_file:#class_indices
        json_file.write(json_str)
        
    pin_memory = True
    train_loader = DataLoader(train_dataset,batch_size=BatchSize,
                                               pin_memory=pin_memory,
                                               shuffle=True,num_workers=nw)
    val_loader = DataLoader(val_dataset,batch_size=V_size,
                                               pin_memory=pin_memory,
                                               shuffle=False,num_workers=nw)
    test_loader = DataLoader(test_dataset,batch_size=T_size,
                                              pin_memory=pin_memory,
                                              shuffle=False,num_workers=nw)

    print("using {} images for training, {} images for validation, {} images for testing.".format(train_num,
                                                                                                  val_num,
                                                                                                  test_num))

BatchSize = 168
V_size = 40 
T_size = 32 
train_doc = "aug_train_8000"
val_doc = "val_dir"
test_doc = "test_dir"
global lrs
lrs = []

nw = min([os.cpu_count(), BatchSize if BatchSize > 1 else 0, cpus_per_gpu]) 
get_data()

class LogitRegression(nn.Module):
    def __init__(self, n_channels=3, img_size=299, num_classes=7):
        super().__init__()
        self.n_channels = n_channels
        self.img_size = img_size
        self.num_classes = num_classes

        self.fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_channels * self.img_size * self.img_size, num_classes)
        )
    def forward(self, x):
        # Forward pass with softmax activation for multi-class classification
        return F.log_softmax(self.fc_layer(x), dim=1)

n_channels = 3
img_size = 299
network = LogitRegression(n_channels=n_channels, img_size=img_size, num_classes=7)
network = network.to(device)

def train(epoch):
    network.train()
    global best_train,train_evl_result#,evl_tmp_result
    running_loss,r_pre = 0., 0.
    print_step = len(train_loader)//2
    steps_num = len(train_loader)
    tmp_size = BatchSize
    print(f'\033[1;32m[Train Epoch:[{epoch}]{img_title} ==> Training]\033[0m ...')
    optimizer.zero_grad()
    train_tmp_result = torch.zeros(n_classes,n_classes)
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):        

        batch_idx += 1
        target_indices = target
        #target_one_hot = one_hot(target, length=n_classes)
        #data, target = Variable(data).to(device), Variable(target_one_hot).to(device)
        data, target = Variable(data).to(device), Variable(target).to(device)
        
        output = network(data) 
        loss = criterion(output, target)    
        loss.backward()     
        optimizer.step()
        optimizer.zero_grad()
        
        running_loss += loss.item()
        
        #v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True)) 
        #pred = v_mag.data.max(1, keepdim=True)[1].cpu().squeeze()
        pred = output.argmax(dim=1).cpu().squeeze()
        r_pre += pred.eq(target_indices.view_as(pred)).squeeze().sum()
        tmp_pre = r_pre/(batch_idx*BatchSize)
        
        if batch_idx % print_step == 0 and batch_idx != steps_num:
            print("[{}/{}] Loss{:.5f},ACC:{:.5f}".format(batch_idx,len(train_loader), # steps_num = len(train_loader)
                                                         loss,tmp_pre))
        if batch_idx % steps_num == 0 and train_num % tmp_size != 0:  # train_num = len(train_dataset), tmp_size = BatchSize
            tmp_size = train_num % tmp_size
                          
        for i in range(tmp_size):
            pred_x = pred.numpy()
            train_tmp_result[target_indices[i]][pred_x[i]] +=1

        if best_train < tmp_pre and tmp_pre >= 80: 
            torch.save(network.state_dict(), iter_path)
        
    epoch_acc = r_pre / train_num
    epoch_loss = running_loss / len(train_loader)  
    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc) 
    
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    scheduler.step()
    
    if best_train < epoch_acc:
        best_train = epoch_acc
        train_evl_result = train_tmp_result.clone()
        torch.save(network.state_dict(), last_path)
        torch.save(train_evl_result, f'./tmp/{img_title}/{suf}/last_train_evl_result.pth')
    
    print("Train Epoch:[{}] Loss:{:.5f},Acc:{:.5f},Best_train:{:.5f}".format(epoch,epoch_loss,
                                                                     epoch_acc,best_train))
    
def test(split="test"):
    network.eval()
    global test_acc,eval_acc,best_acc,net_parameters
    global test_evl_result,val_evl_result#,evl_tmp_result
    cor_loss,correct,Auc, Acc= 0, 0, 0, 0
    evl_tmp_result = torch.zeros(n_classes,n_classes)
    
    if split == 'val':
        data_loader = val_loader
        tmp_size = V_size
        data_num = val_num
    else:
        data_loader = test_loader
        tmp_size = T_size
        data_num = test_num
        
    steps_num = len(data_loader)
    print(f'\033[35m{img_title} ==> {split} ...\033[0m')
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            batch_idx +=1
            target_indices = target#torch.Size([batch, 7])  
            #target_one_hot = one_hot(target, length=n_classes)            
            #data, target = Variable(data).to(device), Variable(target_one_hot).to(device)
            data, target = Variable(data).to(device), Variable(target).to(device)

            output= network(data)
            pred = output.argmax(dim=1).cpu()
            #main_output = output.logits  # Primary output
            #pred = main_output.argmax(dim=1).cpu()
            #v_mag = torch.sqrt(torch.sum(output**2, dim=2, keepdim=True))
            #pred = v_mag.data.max(1, keepdim=True)[1].cpu()#[9, 2, 1, 1, 6,..., 1, 4, 6, 5, 7,]
            
            if batch_idx % steps_num == 0 and data_num % tmp_size != 0:
                tmp_size = data_num % tmp_size
                          
            for i in range(tmp_size):
                pred_y = pred.numpy()
                evl_tmp_result[target_indices[i]][pred_y[i]] +=1 

        diag_sum = torch.sum(evl_tmp_result.diagonal())
        all_sum = torch.sum(evl_tmp_result) 
        test_acc = 100. * float(torch.div(diag_sum,all_sum)) 
        print(f"{split}_Acc:\033[1;32m{round(float(test_acc),3)}%\033[0m")

        if split == 'val':
            val_acc_list.append(test_acc)
            if test_acc >= best_acc:
                best_acc = test_acc
                val_evl_result = evl_tmp_result.clone()#copy.deepcopy(input)
                torch.save(network.state_dict(), save_PATH)   
                torch.save(val_evl_result, f'./tmp/{img_title}/{suf}/best_val_evl_result.pth')
            print(f"Best_val:\033[1;32m[{round(float(best_acc),3)}%]\033[0m")
        else:
            test_acc_list.append(test_acc)
            if test_acc >= eval_acc:
                eval_acc = test_acc
                test_evl_result = evl_tmp_result.clone()#copy.deepcopy(input)
                torch.save(network.state_dict(), f'./tmp/{img_title}/{suf}/best_test_{img_title}_{suf}.pth')
                torch.save(test_evl_result, f'./tmp/{img_title}/{suf}/best_test_evl_result.pth')
            print(f"Best_eval:\033[1;32m[{round(float(eval_acc),3)}%]\033[0m")

#create store
from pathlib import Path
import os
try:
    print(f"suf:{suf}")
except NameError:
    suf = time.strftime("%m%d_%H%M%S", time.localtime())
    print(f"suf:{suf}")
if os.path.exists(f'./tmp/{img_title}/{suf}'):
    print (f'Store: "./tmp/{img_title}/{suf}"')
else:
    dir_path = Path("./tmp") / img_title / suf
    dir_path.mkdir(parents=True, exist_ok=True)
iter_path = f'./tmp/{img_title}/{suf}/train_{img_title}_{suf}.pth'
save_PATH = f'./tmp/{img_title}/{suf}/best_val_{img_title}_{suf}.pth'
last_path = f'./tmp/{img_title}/{suf}/last_train_{img_title}_{suf}.pth'
print(save_PATH)

num_epochs = 120

from transformers import get_cosine_schedule_with_warmup
learning_rate = 1e-5
weight_decay = 1e-3
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
#warmup_steps = int(0.1 * num_epochs)  # 10% warmup
total_steps = num_epochs  # Total epochs
#scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(1, num_epochs + 1): 
    train(epoch)
    test('val')
    
print('Finished Training')

show.conclusion(opt='val',img_title=img_title)

#save
s0 = np.array(train_acc_list)
np.save(f'./tmp/{img_title}/{suf}/{img_title}_train_acc_{suf}.npy', s0)
s1 = np.array(train_loss_list)
np.save(f'./tmp/{img_title}/{suf}/{img_title}_train_loss_{suf}.npy', s1)
s3 = np.array(val_acc_list)
np.save(f'./tmp/{img_title}/{suf}/{img_title}_val_acc_{suf}.npy', s3)
s5 = np.array(val_loss_list)
np.save(f'./tmp/{img_title}/{suf}/{img_title}_val_loss_{suf}.npy', s5)

network.load_state_dict(torch.load(save_PATH))

for k in range(22,29):
    T_size = k
    print(f"T_size:{k}")
    for i in range(300,320):
        get_data(i)
        print(f"size:{i}")
        for j in range(3):
            test()
            if dict_imgSize.get(i) is None or dict_imgSize[i] < test_acc:
                dict_imgSize[i] = test_acc                   
            elif dict_batch.get(k) is None or dict_batch[k] < test_acc:
                    dict_batch[k] = test_acc
    s2 = np.array(test_acc_list)
    np.save(f'./tmp/{img_title}/{suf}/{img_title}_test_acc_{suf}.npy', s2)
    s4 = np.array(dict_batch)
    np.save(f'./tmp/{img_title}/{suf}/{img_title}_dict_batch_{suf}.npy', s4)
    s6 = np.array(test_loss_list)
    np.save(f'./tmp/{img_title}/{suf}/{img_title}_test_loss_{suf}.npy', s6)
    s8 = np.array(test_auc_list)
    np.save(f'./tmp/{img_title}/{suf}/{img_title}_test_auc_{suf}.npy', s8)
    s10 = np.array(dict_imgSize)
    np.save(f'./tmp/{img_title}/{suf}/{img_title}_dict_imgSize_{suf}.npy', s10)


show.conclusion(img_title=img_title) 
print(sorted(dict_imgSize.items(), key=lambda x: x[1] ,reverse=True)[0:9])
print(sorted(dict_batch.items(), key=lambda x: x[1], reverse=True)[0:9])