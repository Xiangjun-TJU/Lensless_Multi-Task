import torch
from torch.autograd import Variable
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
from datetime import datetime
#from lib.pvt8 import PolypPVT1
from utils0.dataloader import get_loader, test_dataset
from utils0.utils import clip_gradient, adjust_lr, AvgMeter
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import logging
from utils1 import SalEval, AverageMeterSmooth, Logger, plot_training_process
import datasets
import transforms
from skimage import io
import scipy.io as sio
from torch.nn import init
import matplotlib.image as mp
from torch_vgg import Vgg16
import argparse
import os
from collections import OrderedDict
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import torch.optim as optim
import yaml
#from albumentations.augmentations import transforms
#from albumentations.core.composition import Compose, OneOf
#from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
#from tqdm import tqdm
#from albumentations import RandomRotate90,Resize
import archs8
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool
from archs import UNext
from focal_frequency_loss import FocalFrequencyLoss as FFL

import matplotlib.pyplot as plt

def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)

    return (wbce + wiou).mean()

@torch.no_grad()
def test(model, Inversion, val_loader):
    salEvalVal = SalEval()
  #  data_path = os.path.join(path, dataset)
  #  image_root = '{}/images/'.format(data_path)
  #  gt_root = '{}/masks/'.format(data_path)
    model.eval()
    Inversion.eval()
   # num1 = len(os.listdir(gt_root))
  #  test_loader = test_dataset(image_root, gt_root, 352)
    DSC = 0.0
    
    for i, pack in enumerate(val_loader, start=1):

        images, gts, meas, body, detail = pack

        images = Variable(images).cuda()
        tg = Variable(gts).cuda()
        meas = Variable(meas).cuda()

        x_init, x_final = Inversion(meas)
     #   x_final = F.interpolate(meas,images.size()[2:])
        Pr, pred = model(x_final)

        # eval Dice
        output_mask = F.upsample(pred, size=tg[0,0].shape, mode='bilinear', align_corners=False)
      #  output_mask = output_mask
        output_mask = output_mask.sigmoid()
        res = output_mask.data.cpu().numpy().squeeze()
        res = (res-np.min(res))/(np.max(res)-np.min(res)+1e-16)
        input = res
        target = np.array(tg.cpu().detach().numpy())
        N = tg.shape
        smooth = 1
        input_flat = np.reshape(input, (-1))
        target_flat = np.reshape(target, (-1))
        intersection = (input_flat * target_flat)
        dice = (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        dice = '{:.4f}'.format(dice)
        dice = float(dice)
        DSC = DSC + dice
        
        
        if epoch>=0:
            x1 = (res*255).astype(np.uint8)
            output_mask_dir = '/home/yxj001/YXJ001/UNeXt-pytorch-main/imgs72/Pred_mask/'
            if not os.path.exists(output_mask_dir):
                os.mkdir(output_mask_dir)
            io.imsave(output_mask_dir+str(i)+'Sal-Maps.png',x1)    
            x2 = Pr[0, :, :, :].squeeze().permute(1,2,0).cpu().detach().numpy()
            x2 = (x2 - np.min(x2))/(np.max(x2) - np.min(x2))
            output_mask_dir = '/home/yxj001/YXJ001/UNeXt-pytorch-main/imgs72/Image/'
            if not os.path.exists(output_mask_dir):
                os.mkdir(output_mask_dir)
            io.imsave(output_mask_dir+str(i)+'Img.png',(x2*255).astype(np.uint8))          
        salEvalVal.addBatch(output_mask[:, 0, :, :], tg[:,0,:,:].bool())
    F_beta, MAE = salEvalVal.getMetric()
  #  record['val']['F_beta'].append(F_beta)
 #   record['val']['MAE'].append(MAE)
    return DSC / i, F_beta, MAE



def train(train_loader, model, Inversion, tv, optimizer, epoch, val_loader):
    salEvalTrain = SalEval()
    model.train()
    Inversion.train()

    size_rates = [0.75, 1, 1.25] 
    loss_P_record = AvgMeter()
    loss_r_record = AvgMeter()
    loss_percep_record = AvgMeter()    
    for i, pack in enumerate(train_loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            # ---- data prepare ----
        #    images, gts = pack
            images, gts, meas, body, detail = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()
            meas = Variable(meas).cuda()
            body = Variable(body).cuda()
            detail = Variable(detail).cuda()
            # ---- rescale ----
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
         #   print('trainsize:',trainsize)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                body = F.upsample(body, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                detail = F.upsample(detail, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            # ---- forward ----
            x_init, x_final = Inversion(meas)
          #  x_final = F.interpolate(meas,images.size()[2:])
            if rate != 1:
                x_final = F.upsample(x_final, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
            Pr, Ps = model(x_final)
##            print('Pr',Pr.shape)
##            print('Ps',Ps.shape)
##            print('images',images.shape)                        
            # ---- loss function ----
            valfeatures_y = vgg(images)				#MSE loss+w*perception loss for deep recon
            valfeatures_x = vgg(Pr)
            a21 = torch.mean(torch.pow((valfeatures_y.relu2_2-valfeatures_x.relu2_2),2))
            a22 = torch.mean(torch.pow((valfeatures_y.relu4_3-valfeatures_x.relu4_3),2))

            loss_discrepancy1 = torch.mean(torch.pow((Pr - images), 2))  #MSEloss
            loss_discrepancy2 = a21+a22
            loss_ffl =  ffl(Pr,images)
         ##   loss_TV = tv(x_final)
#            print('Ps_f',Ps_f.shape)           
#            print('gts',gts.shape)
#            loss_P1 = structure_loss(P1, detail)
#            loss_P2 = structure_loss(P2, body)
            loss_P = structure_loss(Ps, gts) 
            loss_r= loss_discrepancy1 + 0.5*loss_discrepancy2
            loss = loss_P + 1.5*loss_r + 0.5*loss_ffl
#            loss = loss_r


            # + 0.5*loss_discrepancy1  + 0.25*loss_discrepancy2 + 0.5*loss_TV
            # ---- backward ----
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            # ---- recording loss ----
            if rate == 1:
                loss_P_record.update(loss_P.data, opt.batchsize)
                loss_r_record.update(loss_r.data, opt.batchsize)
                loss_percep_record.update(loss_discrepancy1.data, opt.batchsize)

        # ---- train visualization ----
        if i % 20 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                  ' lateral-5: {:0.4f}, loss_r: {:0.4f},' 'loss_mse: {:0.4f}]' .
                  format(datetime.now(), epoch, opt.epoch, i, total_step,
                         loss_P_record.show(),loss_r_record.show(),loss_percep_record.show()))
    # save model 
    save_path = (opt.train_save)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT.pth')
    torch.save(Inversion.state_dict(), save_path +str(epoch)+ 'Inv.pth')
    # choose the best model

    global dict_plot
    global best1
    global best2
    global best3
    if (epoch + 1) % 1 == 0:
        for dataset in ['CVC-300']:
            dataset_dice, F_beta, MAE = test(model, Inversion, val_loader)
            logging.info('epoch: {}, dataset: {}, dice: {},F_beta:{}, MAE:{}'.format(epoch, dataset, dataset_dice,F_beta, MAE))
            print(dataset, 'dataset_dice: ', dataset_dice)
            print(dataset, 'F_beta: ', F_beta)
            print(dataset, 'MAE: ', MAE)
            dict_plot[dataset].append(dataset_dice)
        meandice, F_beta, MAE = test(model, Inversion, val_loader)
        dict_plot['test'].append(meandice)
        if meandice > best1:
            best1 = meandice
            torch.save(model.state_dict(), save_path + 'PolypPVTa.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best1.pth')
            torch.save(Inversion.state_dict(), save_path +str(epoch)+ 'Inv-best1.pth')
            print('##############################################################################Dice_best', best1)
            logging.info('##############################################################################Dice_best:{}'.format(best1))
        if F_beta > best2:
            best2 = F_beta
            torch.save(model.state_dict(), save_path + 'PolypPVTb.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best2.pth')
            torch.save(Inversion.state_dict(), save_path +str(epoch)+ 'Inv-best2.pth')
            print('##############################################################################F_beta_best', best2)
            logging.info('##############################################################################F_beta_best:{}'.format(best2))
        if MAE > best3:
            best3 = MAE
            torch.save(model.state_dict(), save_path + 'PolypPVTc.pth')
            torch.save(model.state_dict(), save_path +str(epoch)+ 'PolypPVT-best3.pth')
            torch.save(Inversion.state_dict(), save_path +str(epoch)+ 'Inv-best3.pth')
            print('##############################################################################MAE_best', best3)
            logging.info('##############################################################################MAE_best:{}'.format(best3))

def plot_train(dict_plot=None, name = None):
    color = ['red', 'lawngreen', 'lime', 'gold', 'm', 'plum', 'blue']
    line = ['-', "--"]
    for i in range(len(name)):
        plt.plot(dict_plot[name[i]], label=name[i], color=color[i], linestyle=line[(i + 1) % 2])
        transfuse = {'CVC-300': 0.902, 'CVC-ClinicDB': 0.918, 'Kvasir': 0.918, 'CVC-ColonDB': 0.773,'ETIS-LaribPolypDB': 0.733, 'test':0.83}
        plt.axhline(y=transfuse[name[i]], color=color[i], linestyle='-')
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.title('Train')
    plt.legend()
    plt.savefig('eval.png')
    # plt.show()
    
def load_meas_matrix():
    WL = np.zeros((500,256,1))
    WR = np.zeros((620,256,1))
    d = sio.loadmat('/home/yxj001/sp0722/Flatnet_quantization-master/flatcam_prototype2_calibdata.mat') ##Initialize the weight matrices with transpose
    phil = np.zeros((500,256,1))
    phir = np.zeros((620,256,1))

    pl = sio.loadmat('/home/yxj001/sp0722/Flatnet_quantization-master/phil_toep_slope22.mat')
    pr = sio.loadmat('/home/yxj001/sp0722/Flatnet_quantization-master/phir_toep_slope22.mat')
    WL[:,:,0] = pl['phil'][:,:,0]
    WR[:,:,0] = pr['phir'][:,:,0] 
    #if args.init_matrix_type=='':

    WL = WL.astype('float32')   #  Pseudo inverse   WL
    WR = WR.astype('float32')   #  Pseudo inverse   WR  

    phil[:,:,0] = d['P1gb']
    phir[:,:,0] = d['Q1gb']
    phil = phil.astype('float32')   #  phiL
    phir = phir.astype('float32')   #  phiR

    return phil,phir,WL,WR

class initial_inversion2(nn.Module):
    def __init__(self):
        super(initial_inversion2, self).__init__()
    def forward(self,meas,WL,WR):
        x0=F.leaky_relu(torch.matmul(torch.matmul(meas[:,0,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x1=F.leaky_relu(torch.matmul(torch.matmul(meas[:,1,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        x2=F.leaky_relu(torch.matmul(torch.matmul(meas[:,2,:,:],WR[:,:,0]).permute(0,2,1),WL[:,:,0]).permute(0,2,1).unsqueeze(3))
        X_init=torch.cat((x0,x1,x2),3)
        X_init = X_init.permute(0,3,1,2)
        return X_init

# Define ISTA-Net
class Inv(torch.nn.Module):
    def __init__(self, phil, phir, WL, WR, LayerNo):
        super(Inv, self).__init__()
        onelayer = []
        self.LayerNo = LayerNo

        self.WL = nn.Parameter(torch.tensor(WL))
        self.WR = nn.Parameter(torch.tensor(WR))
        self.ini_inversion = initial_inversion2()
        
    def forward(self, meas):
        x = self.ini_inversion(meas,self.WL,self.WR)
        savename = 'phil_epoch' 
        np.save(savename, self.WL.detach().cpu().numpy())
        savename = 'phir_epoch'
        np.save(savename, self.WR.detach().cpu().numpy())   
        x_init = x
        
        return [x_init, x]
class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        print("You are using `CEL`!")
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)
class W_BCE_IOU_loss(nn.Module):
    def __init__(self):
        super(W_BCE_IOU_loss, self).__init__()
    def forward(self, pred, mask):
        weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
        wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
        wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

        pred  = torch.sigmoid(pred)
        inter = ((pred*mask)*weit).sum(dim=(2,3))
        union = ((pred+mask)*weit).sum(dim=(2,3))
        wiou  = 1-(inter+1)/(union-inter+1)
        return (wbce+wiou).mean()

class BCE_IOU_loss(nn.Module):
    def __init__(self):
        super(BCE_IOU_loss, self).__init__()

    def forward(self, inputs, target):
       # target = target.float()
        loss = F.binary_cross_entropy(inputs, target) + iou_loss(inputs, target) 
        return loss.mean()
class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()
def mae_loss(pred, mask):
    return torch.mean(torch.abs(pred-mask))            
if __name__ == '__main__':
    dict_plot = {'CVC-300':[], 'CVC-ClinicDB':[], 'Kvasir':[], 'CVC-ColonDB':[], 'ETIS-LaribPolypDB':[], 'test':[]}
    name = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'test']
    ##################model_name#############################
    model_name = 'PolypPVT1'
    ###############################################
    parser = argparse.ArgumentParser()

    parser.add_argument('--epoch', type=int,
                        default=100, help='epoch number')

    parser.add_argument('--lr', type=float,
                        default=1e-4, help='learning rate')

    parser.add_argument('--optimizer', type=str,
                        default='AdamW', help='choosing optimizer AdamW or SGD')

    parser.add_argument('--augmentation',
                        default=False, help='choose to do random flip rotation')

    parser.add_argument('--batchsize', type=int,
                        default=12, help='training batch size')

    parser.add_argument('--trainsize', type=int,
                        default=256, help='training dataset size')

    parser.add_argument('--clip', type=float,
                        default=0.5, help='gradient clipping margin')

    parser.add_argument('--decay_rate', type=float,
                        default=0.1, help='decay rate of learning rate')

    parser.add_argument('--decay_epoch', type=int,
                        default=50, help='every n epochs decay learning rate')

    parser.add_argument('--train_path', type=str,
                        default='./dataset/TrainDataset/',
                        help='path to train dataset')

    parser.add_argument('--test_path', type=str,
                        default='./dataset/TestDataset/',
                        help='path to testing Kvasir dataset')

    parser.add_argument('--train_save', type=str,
                        default='./model_pth17ffl/'+model_name+'/')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='UNext')
    parser.add_argument('--deep_supervision', default=True, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')


    opt = parser.parse_args()
    config = vars(opt)

    logging.basicConfig(filename='train_log1.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

    # ---- build models ----
    # torch.cuda.set_device(0)  # set your gpu device
#    model = PolypPVT1().cuda()
    model = archs8.__dict__[config['arch']](config['num_classes'],config['input_channels'],config['deep_supervision'])
    model = model.cuda()
    PhiL, PhiR, WL, WR = load_meas_matrix()
# load the model
    vgg = Vgg16(requires_grad=False)
    ffl = FFL(loss_weight=1.0, alpha=1.0).cuda() 
    Inversion = Inv(PhiL, PhiR, WL, WR, 4).cuda()
    Inversion.load_state_dict(torch.load('/home/yxj001/YXJ001/Paper_first/model/MLP2_Net_layer_DUnet_4_group_1_ratio_25_lr_0.0001/net_params_20.pth'))
##    model.load_state_dict(torch.load('/home/yxj001/YXJ001/Polyp-PVT-main/model_pth2/PolypPVT1/10PolypPVT-best2.pth'))
    criterion = L_TV().cuda()
    Vgg = vgg.cuda()
    best1,best2,best3 = 0,0,0

   # params = 

    if opt.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(list(model.parameters())+list(Inversion.parameters()), opt.lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.SGD(list(model.parameters())+list(Inversion.parameters()), opt.lr, weight_decay=1e-4, momentum=0.9)

    print(optimizer)
    image_root = '{}/images/'.format(opt.train_path)
    gt_root = '{}/masks/'.format(opt.train_path)

#    train_loader = get_loader(image_root, gt_root, batchsize=opt.batchsize, trainsize=opt.trainsize,
#                              augmentation=opt.augmentation)
# so that we can generate more augmented data and prevent the network from overfitting
    train_set = datasets.Dataset('', 'Lensless_Train3', transform=None)
    val_set = datasets.Dataset('', 'Lensless_Test3', transform=None)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, num_workers = 8, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers = 4, pin_memory=True)
    
    total_step = len(train_loader)
    

    print("#" * 20, "Start Training", "#" * 20)
    for epoch in range(1, opt.epoch):
        adjust_lr(optimizer, opt.lr, epoch, 0.1, 200)
#        test(model, Inversion, val_loader)
        train(train_loader, model, Inversion, criterion, optimizer, epoch, val_loader)
    
    # plot the eval.png in the training stage
    # plot_train(dict_plot, name)
