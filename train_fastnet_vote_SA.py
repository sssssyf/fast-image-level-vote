import os
import argparse
import time
import numpy as np
import torch
import datetime
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.metrics import cohen_kappa_score
from func import load,product,intersectionAndUnionGPU,generate_png
from sklearn.decomposition import PCA

## GPU_configration

USE_GPU=True
if USE_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# else:
#     device=torch.device('cpu')
#     print('using device:',device)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(args, epoch, net, optimizer, trn_loader, criterion, categories):
    net.train()  # train mode
    #criterion2=DiceLoss()
    max_iter=args.epochs * len(trn_loader)
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    target_meter = AverageMeter()
    for idx, (X_data, y_target) in enumerate(trn_loader):

        X_data=Variable(X_data.float()).cuda(non_blocking=True)
        y_target = Variable(y_target.float().long()).cuda(non_blocking=True)
        y_pred = net.forward(X_data)
        _, predicted = torch.max(y_pred[-1], 1)  #投票策略

        loss = criterion(y_pred[-1],y_target)+criterion(y_pred[0],y_target)+criterion(y_pred[1],y_target)+criterion(y_pred[2],y_target)+criterion(y_pred[3],y_target)   #联合监督

        # back propagation
        optimizer.zero_grad()
        if args.use_apex=='True':
            with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        # compute acc
        n = X_data.size(0)  # batch size
        loss_meter.update(loss.item(), n)
        intersection, _, target = intersectionAndUnionGPU(predicted, y_target, categories-1, args.ignore_label)
        intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), target_meter.update(target)


        del X_data, y_target
        del y_pred

        # updata lr
        if args.resume:
            current_lr = args.lr
        else:
            current_iter = epoch * len(trn_loader) + idx + 1
            current_lr = args.lr * (1 - float(current_iter) / max_iter) ** 0.9

        optimizer.param_groups[0]['lr'] = current_lr

        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mAcc = np.mean(accuracy_class)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # oa of a bs

        print('Training epoch [{}/{}]: Loss {:.4f} AA/OA {:.4f}/{:.4f}.'.format(epoch + 1,
                                                                args.epochs,loss_meter.avg,
                                                                mAcc,accuracy))

def validation(args, epoch, net, val_loader, categories):
    print('>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<')
    net.eval()  # evaluation mode
    intersection_meter = AverageMeter()
    target_meter = AverageMeter()
    for idx, (X_data, y_target) in enumerate(val_loader):
        with torch.no_grad():
            X_data = Variable(X_data.float()).cuda(non_blocking=True)
            y_target = Variable(y_target.float().long()).cuda(non_blocking=True)
            y_pred = net.forward(X_data)

        _, predicted = torch.max(y_pred, 1)


        # compute acc
        intersection, _, target = intersectionAndUnionGPU(predicted, y_target, categories - 1, args.ignore_label)
        intersection, target = intersection.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)  # oa of a bs


        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
        mAcc = np.mean(accuracy_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    print('Validation epoch [{}/{}]: AA/OA {:.4f}/{:.4f}.'.format(epoch + 1,
                                                                args.epochs, mAcc, allAcc))
    for i in range(categories-1):
        print('Class_{}: accuracy {:.4f}.'.format(i+1, accuracy_class[i]))

    print('>>>>>>>>>>>>>>>> End Evaluation <<<<<<<<<<<<<<<<<<')

    return allAcc


def resolution_compression(X_data,y_trn_data,y_gt_data, h_ratio,w_ratio):
    print('>>>>>>>>>>>>>>>> Start Compressing Resolutopm <<<<<<<<<<<<<<<<<<')
    new_h = int(X_data.shape[2] / h_ratio)
    new_w = int((X_data.shape[3] - X_data.shape[3] % w_ratio) / w_ratio)

    X_data_split_h = torch.empty(h_ratio, X_data.shape[1], new_h, X_data.shape[3])
    y_trn_data_split_h = torch.empty(h_ratio, new_h, X_data.shape[3])
    y_gt_data_split_h = torch.empty((h_ratio, new_h, X_data.shape[3]), dtype=torch.int32)

    X_data_split_h_w = torch.empty(h_ratio * w_ratio, X_data.shape[1], new_h, new_w)
    y_trn_data_split_h_w = torch.empty(h_ratio * w_ratio, new_h, new_w)
    y_gt_data_split_h_w = torch.empty((h_ratio * w_ratio, new_h, new_w), dtype=torch.int32)

    for j in range(h_ratio):
        for i in range(new_h):
            if i == 0:
                x = X_data[:, :, h_ratio * i + j, :].reshape(1, X_data.shape[1], 1, X_data.shape[3])
                y = y_trn_data[:, h_ratio * i + j, :].reshape(1, 1, X_data.shape[3])
                z = y_gt_data[:, h_ratio * i + j, :].reshape(1, 1, X_data.shape[3])
            else:
                x = torch.cat((x, X_data[:, :, h_ratio * i + j, :].reshape(1, X_data.shape[1], 1, X_data.shape[3])),
                              dim=2)
                y = torch.cat((y, y_trn_data[:, h_ratio * i + j, :].reshape(1, 1, X_data.shape[3])), dim=1)
                z = torch.cat((z, y_gt_data[:, h_ratio * i + j, :].reshape(1, 1, X_data.shape[3])), dim=1)
        X_data_split_h[j, :, :, :] = x
        y_trn_data_split_h[j, :, :] = y
        y_gt_data_split_h[j, :, :] = z

    for k in range(h_ratio):
        for j in range(w_ratio):
            for i in range(new_w):
                if i == 0:
                    x = X_data_split_h[k, :, :, w_ratio * i + j].reshape(1, X_data_split_h.shape[1],
                                                                         X_data_split_h.shape[2], 1)
                    y = y_trn_data_split_h[k, :, w_ratio * i + j].reshape(1, X_data_split_h.shape[2], 1)
                    z = y_gt_data_split_h[k, :, w_ratio * i + j].reshape(1, X_data_split_h.shape[2], 1)
                else:
                    x = torch.cat((x, X_data_split_h[k, :, :, w_ratio * i + j].reshape(1, X_data_split_h.shape[1],
                                                                                       X_data_split_h.shape[2], 1)),
                                  dim=3)
                    y = torch.cat((y, y_trn_data_split_h[k, :, w_ratio * i + j].reshape(1, X_data_split_h.shape[2], 1)),
                                  dim=2)
                    z = torch.cat((z, y_gt_data_split_h[k, :, w_ratio * i + j].reshape(1, X_data_split_h.shape[2], 1)),
                                  dim=2)

            X_data_split_h_w[k * h_ratio + j, :, :, :] = x
            y_trn_data_split_h_w[k * h_ratio + j, :, :] = y
            y_gt_data_split_h_w[k * h_ratio + j, :, :] = z

    print('>>>>>>>>>>>>>>>> End Compressing <<<<<<<<<<<<<<<<<<')
    return X_data_split_h_w,y_trn_data_split_h_w, y_gt_data_split_h_w,new_h,new_w


def main():
    ############ parameters setting ############

    parser = argparse.ArgumentParser(description="Network Trn_val_Tes")
    ## dataset setting
    parser.add_argument('--dataset', type=str, default='salina',
                        choices=['indian','pavia','houston','salina','ksc','paviaC'],
                        help='dataset name')
    ## network setting
    parser.add_argument('--network', type=str, default='fastnet_vote',
                        choices=['FContNet','DeepLabV3Plus','SETR','SegNet','UNET','CBAM','octave','freenet','fastnet_vote'],
                        help='network name')
    ## normalization setting
    parser.add_argument('--norm', type=str, default='std',
                        choices=['std','norm'],
                        help='nomalization mode')
    parser.add_argument('--mi', type=int, default=-1,
                        help='min normalization range')
    parser.add_argument('--ma', type=int, default=1,
                        help='max normalization range')
    ## experimental setting
    parser.add_argument('--use_apex', type=str, default='false',
                        choices=['True', 'False'],help='mixed-precision training')
    parser.add_argument('--opt_level', type=str, default='O1',
                        choices=['O0', 'O1','O2'], help='mixed-precision')
    parser.add_argument('--input_mode', type=str, default='whole',
                        choices=['whole', 'part'],help='input setting')
    parser.add_argument('--input_size', nargs='+',default=128, type=int)
    parser.add_argument('--overlap_size', type=int, default=16,
                        help='size of overlap')
    parser.add_argument('--experiment-num', type=int, default=10,
                        help='experiment trials number')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=1,
                        help='input batch size for validation')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='weight decay')
    parser.add_argument('--workers', type=int, default=0,
                        help='workers num')
    parser.add_argument('--ignore_label', type=int, default=255,
                        help='ignore label')
    parser.add_argument('--print_freq', type=int, default=3,
                        help='print frequency')
    parser.add_argument("--resume", type=str, help="model path.")
    ## model setting
    parser.add_argument('--mode', type=str, default='p_c_s',
                        choices=['p_c_s', 'c_p_s', 'p+c_s', 'p_s_c'], help='context sequence')
    parser.add_argument('--head', type=str, default='psp',
                        choices=['psp','aspp'],help='seghead')


    args = parser.parse_args()

    ############# load dataset(indian_pines & pavia_univ...)######################

    a=load()

    All_data,labeled_data,rows_num,categories,r,c,FLAG=a.load_data(flag=args.dataset)
    all_num=All_data[:,0]
    print('Data has been loaded successfully!')

    ##################### normlization ######################

    if args.norm=='norm':
        scaler = MinMaxScaler(feature_range=(args.mi,args.ma))
        All_data_norm=scaler.fit_transform(All_data[:,1:-1])

    elif args.norm=='std':
        scaler = StandardScaler()
        All_data_norm = scaler.fit_transform(All_data[:, 1:-1])    #归一化

    print('Image normlization successfully!')

    ########################### Data preparation ##################################

    if args.input_mode=='whole':

        X_data=All_data_norm.reshape(1,r,c,-1)

        args.print_freq=1

        args.input_size = (r, c)

    elif args.input_mode=='part':


        image_size=(r, c)

        input_size=args.input_size

        LyEnd,LxEnd = np.subtract(image_size, input_size)

        Lx = np.linspace(0, LxEnd, np.ceil(LxEnd/np.float(input_size[1]-args.overlap_size))+1, endpoint=True).astype('int')
        Ly = np.linspace(0, LyEnd, np.ceil(LyEnd/np.float(input_size[0]-args.overlap_size))+1, endpoint=True).astype('int')

        image_3D=All_data_norm.reshape(r,c,-1)

        N=len(Ly)*len(Lx)

        X_data=np.zeros([N,input_size[0],input_size[1],image_3D.shape[-1]])#N,H,W,C

        i=0
        for j in range(len(Ly)):
            for k in range(len(Lx)):
                rStart,cStart = (Ly[j],Lx[k])
                rEnd,cEnd = (rStart+input_size[0],cStart+input_size[1])
                X_data[i] = image_3D[rStart:rEnd,cStart:cEnd,:]
                i+=1
    else:
        raise NotImplementedError

    print('{} image preparation Finished!, Data Number {}, '
          'Data size ({},{})'.format(args.dataset,X_data.shape[0],X_data.shape[1],X_data.shape[2]))



    X_data = torch.from_numpy(X_data.transpose(0, 3, 1, 2))#N,C,H,W

    ##################################### trn/val/tes ####################################

    #Experimental memory
    Experiment_result=np.zeros([categories+4,args.experiment_num+2])#OA,AA,kappa,trn_time,tes_time

    #kappa
    kappa=0

    y_map=All_data[:, -1].reshape(r,c)

    y_gt_data=y_map.reshape(1,r,c)
    y_gt_data = torch.from_numpy(y_gt_data)


    print('Implementing FcontNet in {} mode with {} head!'.format(args.mode,args.head))

    for count in range(0, args.experiment_num):


        #################################### trn_label #####################################
        a = product(c, FLAG, All_data)

        rows_num, trn_num, val_num, tes_num, pre_num = a.generation_num(labeled_data, rows_num,count)

        y_trn_map=a.production_label(trn_num, y_map, split='Trn')


        #ground_truth = spectral.imshow(classes=y_trn_map.astype(int), figsize=(9, 9))
       # plt.pause(20)
        # plt.xlabel('trn_label_map')
        # plt.imshow(y_trn_map,cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        if args.input_mode == 'whole':

            y_trn_data=y_trn_map.reshape(1,r,c)

        elif args.input_mode=='part':

            y_trn_data = np.zeros([N, input_size[0], input_size[1]], dtype=np.int32)  # N,H,W

            i=0
            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = Ly[j], Lx[k]
                    rEnd, cEnd = rStart + input_size[0], cStart + input_size[1]
                    y_trn_data[i] = y_trn_map[rStart:rEnd, cStart:cEnd]
                    i+=1
        else:
            raise NotImplementedError


        # plt.xlabel('trn_data_map')
        # plt.imshow(y_trn_data[0], cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        y_trn_data-=1

        y_trn_data[y_trn_data<0]=255

        y_trn_data = torch.from_numpy(y_trn_data)

        print('Experiment {}，training dataset preparation Finished!'.format(count))

        #################################### val_label #####################################

        y_val_map = a.production_label(val_num, y_map, split='Val')

        # plt.xlabel('val_label_map')
        # plt.imshow(y_val_map, cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        if args.input_mode == 'whole':

            y_val_data = y_val_map.reshape(1, r, c)

        elif args.input_mode == 'part':

            y_val_data = np.zeros([N, input_size[0], input_size[1]])  # N,H,W

            i=0
            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = (Ly[j], Lx[k])
                    rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                    y_val_data[i,:,:] = y_val_map[rStart:rEnd, cStart:cEnd]
                    i+=1
        else:
            raise NotImplementedError

        # plt.xlabel('val_data_map')
        # plt.imshow(y_val_data[0], cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        y_val_data -= 1

        y_val_data[y_val_data < 0] = 255

        y_val_data = torch.from_numpy(y_val_data)

        print('Experiment {}，validation dataset preparation Finished!'.format(count))

        ########## training/Validation #############

        torch.cuda.empty_cache()#GPU memory released

        h_ratio = 1
        w_ratio = 1

        if h_ratio > 1 or w_ratio > 1 :

            X_data_split_h_w, y_trn_data_split_h_w, y_gt_data_split_h_w,new_h,new_w = resolution_compression(X_data,y_trn_data,y_gt_data,h_ratio,w_ratio)
            trn_dataset = TensorDataset(X_data_split_h_w, y_trn_data_split_h_w)

            trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                shuffle=True, drop_last=True, pin_memory=True)
        else:
            X_data=X_data[:, :, :,:208]
            y_trn_data=y_trn_data[:,:,:208]
            trn_dataset = TensorDataset(X_data, y_trn_data)
            trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, num_workers=args.workers,
                                    shuffle=True, drop_last=True, pin_memory=True)


        if args.network == 'freenet':
            from models.freenet import freenet#oct_resnet50
            net = freenet(in_ch=X_data.shape[1], num_classes=categories-1)
        elif args.network == 'fastnet_vote':
            from models.fastnet_vote import get_cls_net#oct_resnet50
            net = get_cls_net(in_ch=X_data.shape[1], num_classes=categories-1)

        else:
            raise NotImplementedError

        params = [dict(params=net.parameters(), lr=args.lr)]

        optimizer = torch.optim.SGD(params, momentum=0.9,
                                    lr=args.lr, weight_decay=args.weight_decay)

        if args.use_apex=='True':# use apex

            net, optimizer = apex.amp.initialize(net.cuda(), optimizer, opt_level=args.opt_level)

        #net= torch.nn.DataParallel(net.cuda())

        net.cuda()

        #patch_replication_callback(net)

        criterion = torch.nn.CrossEntropyLoss(ignore_index=args.ignore_label)

        trn_time=0
        best_val_OA=0

        if args.resume:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location='cpu')
                print("=> loading ft model...")
                ckpt_dict = checkpoint.state_dict()
                model_dict = {}
                state_dict = net.state_dict()
                for k, v in ckpt_dict.items():
                    if k in state_dict:
                        model_dict[k] = v
                state_dict.update(model_dict)
                net.load_state_dict(state_dict)

                print("=> loaded checkpoint '{}' ".format(args.resume))
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))
                raise NotImplementedError
        trn_time_one_exp=0
        for i in range(0, args.epochs):
            trn_time1 = time.time()
            train(args, i, net, optimizer, trn_loader, criterion,categories)
            trn_time2 = time.time()
            trn_time_one_exp = trn_time_one_exp+ trn_time2 - trn_time1

            '''
            if (i+1) % 1==0:
                val_OA = validation(args, i, net, val_loader, categories)

                if val_OA >= best_val_OA:
                    filename = str(args.network) + '_' + str(FLAG) + '_' + 'experiment_{}'.format(count) + '_valbest_tmp' + '.pth'
                    torch.save(net, filename)
                    best_val_OA=val_OA
            '''

        print('########### Experiment {}，Model Training Period Finished! ############'.format(count))

        #################################### test_label ####################################

        #y_tes_map = a.production_label(tes_num, y_map, split='Tes')
        y_tes_map = a.production_label(all_num, y_map, split='Tes')

        # plt.xlabel('tes_label_map')
        #
        # plt.imshow(y_tes_map, cmap='jet')
        # plt.xticks([])
        # plt.yticks([])
        #
        # plt.show()

        y_tes_data = y_tes_map.reshape(r, c)

        y_tes_data=y_tes_data[:,:208]

        y_tes_data -= 1


        y_tes_data[y_tes_data < 0] = 255


        print('Experiment {}，Testing dataset preparation Finished!'.format(count))

        ################### testing ################

        #filename = str(args.network) + '_' + str(FLAG) + '_' + 'experiment_{}'.format(count) + '_valbest_tmp' + '.pth'  #加载测试模型
        #net = torch.load(filename, map_location='cpu')
        #net = net.cuda()

        tes_time1 = time.time()

        if args.input_mode == 'whole':

            net.eval()
            with torch.no_grad():
                if h_ratio > 1 or w_ratio > 1:
                    pred=torch.empty((X_data_split_h_w.shape[0],categories-1,X_data_split_h_w.shape[2],X_data_split_h_w.shape[3]))
                    for i in range(X_data_split_h_w.shape[0]):
                        pred[i,:,:,:] = net(X_data_split_h_w[i,:,:,:].reshape(1,X_data_split_h_w.shape[1],X_data_split_h_w.shape[2],X_data_split_h_w.shape[3]).float().cuda())
                else:  pred_vote = net(X_data.float().cuda())
                preds=[]
                for i in range(len(pred_vote)):
                    pred_vote[i]=pred_vote[i].cpu().numpy()
                    pred = np.argmax(pred_vote[i], 1)
                    preds.append(pred)

                preds = np.concatenate(preds, 0)

                import scipy

                y_tes_pred = scipy.stats.mode(preds, axis=0).mode[0]
                y_tes_pred = y_tes_pred.reshape(1,y_tes_pred.shape[0],y_tes_pred.shape[1])
                #y_tes_pred = preds[-1, :, :].reshape(1, preds.shape[1], preds.shape[2])  # 是否投票


                y_tes_pred_map=y_tes_pred


        elif args.input_mode == 'part':

            img=torch.from_numpy(image_3D).permute(2,0,1) #C,H,W
            y_tes_pred = np.zeros([r, c])
            net.eval()

            for j in range(len(Ly)):
                for k in range(len(Lx)):
                    rStart, cStart = (Ly[j], Lx[k])
                    rEnd, cEnd = (rStart + input_size[0], cStart + input_size[1])
                    img_part = img[:,rStart:rEnd,cStart:cEnd].unsqueeze(0)
                    with torch.no_grad():
                        pred = net(img_part.float())
                    pred = pred.cpu().numpy()
                    pred = np.argmax(pred,1).squeeze(0)

                    if j == 0 and k == 0:
                        y_tes_pred[rStart:rEnd, cStart:cEnd] = pred
                    elif j == 0 and k > 0:
                        y_tes_pred[rStart:rEnd, cStart + int(args.overlap_size / 2):cEnd] = pred[:,
                                                                                           int(args.overlap_size / 2):]
                    elif j > 0 and k == 0:
                        y_tes_pred[rStart + int(args.overlap_size / 2):rEnd, cStart:cEnd] = pred[
                                                                                           int(args.overlap_size / 2):,
                                                                                           :]
                    else:
                        y_tes_pred[rStart + int(args.overlap_size / 2):rEnd,
                        cStart + int(args.overlap_size / 2):cEnd] = pred[int(args.overlap_size / 2):,
                                                                    int(args.overlap_size / 2):]
        else:
            raise NotImplementedError

        tes_time2 = time.time()


        print('########### Experiment {}，Model Testing Period Finished! ############'.format(count))

        ####################################### assess ###########################################

        if h_ratio>1 or w_ratio>1:
            y_tes_pred_reshape = y_tes_pred_map.reshape(h_ratio,w_ratio,new_h,new_w)
            y_tes_pred_merge_w = np.zeros((h_ratio,new_h, new_w*w_ratio))
            y_tes_pred_merge_w_h = np.zeros((new_h*h_ratio, new_w*w_ratio))

            for k in range(h_ratio):
                for j in range(w_ratio):
                    for i in range(new_w):
                        y_tes_pred_merge_w[k,:,w_ratio * i + j]=y_tes_pred_reshape[k, j, :, i].reshape(new_h)


            for j in range(h_ratio):
                for i in range(new_h):
                    y_tes_pred_merge_w_h[h_ratio * i + j,:]=y_tes_pred_merge_w[j,i,:].reshape(new_w*w_ratio)


            y_tes_data_1d = y_gt_data_split_h_w.reshape(h_ratio * w_ratio* new_h* new_w).cpu().numpy()

            y_tes_pred_1d = y_tes_pred.reshape(h_ratio * w_ratio* new_h* new_w)

            y_tes_pred_merge_w_h=y_tes_pred_merge_w_h.reshape(-1)
            generate_png(y_tes_data_1d, y_tes_pred_merge_w_h,FLAG,h_ratio*new_h,w_ratio*new_w, all_num) #画图

            rows = np.arange(y_tes_data_1d.shape[0])  # start from 0
            y_tes_data_1d_all = np.c_[rows, y_tes_data_1d]# ID(row number), data, class number
            labeled_data = y_tes_data_1d_all[y_tes_data_1d_all[:, -1] != 0, :] # Removing background and obtain all labeled data
            tes_num =labeled_data[:, 0] # All ID of labeled  data
            y_tes_gt = y_tes_data_1d[tes_num]-1
            y_tes = y_tes_pred_1d[tes_num]

        else:
            y_tes_data_1d = y_tes_data.reshape(y_tes_data.shape[0]*y_tes_data.shape[1])
            y_tes_pred_1d = y_tes_pred.reshape(y_tes_pred.shape[1]*y_tes_pred.shape[2])

            generate_png(y_tes_data_1d, y_tes_pred_1d,FLAG,512,208, all_num) #画图

            y_tes_gt=[]
            y_tes=[]
            for i in range (len(y_tes_data_1d)):
                if y_tes_data_1d[i] != 255:
                    y_tes_gt.append(y_tes_data_1d[i])
                    y_tes.append(y_tes_pred_1d[i])

            y_tes_gt=np.array(y_tes_gt)
            y_tes_gt.astype(int)
            y_tes=np.array(y_tes)-1
            y_tes.astype(int)

        print('==================Test set=====================')
        print('Experiment {}，Testing set OA={}'.format(count,np.mean(y_tes_gt==y_tes)))



        num_tes=np.zeros([categories-1])
        num_tes_pred=np.zeros([categories-1])

        for k in y_tes_gt:
            num_tes[int(k)]+=1# class index start from 0
        for j in range(y_tes_gt.shape[0]):
            if y_tes_gt[j]==y_tes[j]:
                num_tes_pred[int(y_tes_gt[j])]+=1

        Acc=num_tes_pred/num_tes*100

        Experiment_result[0, count]=np.mean(y_tes_gt==y_tes)*100#OA
        Experiment_result[1, count]=np.mean(Acc)#AA
        Experiment_result[2, count]=cohen_kappa_score(y_tes_gt,y_tes)*100#Kappa
        Experiment_result[3, count] = trn_time_one_exp
        Experiment_result[4, count] = tes_time2 - tes_time1
        Experiment_result[5:,count]=Acc

        print('Experiment {}，Testing set AA={}'.format(count, np.mean(Acc)))
        print('Experiment {}，Testing set Kappa={}'.format(count, cohen_kappa_score(y_tes_gt, y_tes)))
        for i in range(categories - 1):
            print('Class_{}: accuracy {:.4f}.'.format(i + 1, Acc[i]))

        print('One time training cost {:.4f} secs'.format(trn_time_one_exp))
        print('One time testing cost {:.4f} secs'.format(tes_time2 - tes_time1))



        print('########### Experiment {}，Model assessment Finished！ ###########'.format(count))

    ########## mean value & standard deviation #############

    Experiment_result[:,-2]=np.mean(Experiment_result[:,0:-2],axis=1)  #计算均值
    Experiment_result[:,-1]=np.std(Experiment_result[:,0:-2],axis=1)   #计算平均差

    print('OA_std={}'.format(Experiment_result[0,-1]))
    print('AA_std={}'.format(Experiment_result[1,-1]))
    print('Kappa_std={}'.format(Experiment_result[2,-1]))
    print('time training cost_std{:.4f} secs'.format(Experiment_result[3,-1]))
    print('time testing cost_std{:.4f} secs'.format(Experiment_result[4,-1]))
    for i in range(Experiment_result.shape[0]):
        if i>4:
            print('Class_{}: accuracy_std {:.4f}.'.format(i-4, Experiment_result[i,-1]))   #均差

    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')


    f = open('./record/'+str(args.network)+'_Time_'+str(day_str)+'_'+str(args.dataset)+'_h_ratio='+str(h_ratio)+'_w_ratio='+str(w_ratio)+'_exp_num='+str(args.experiment_num)+'.txt', 'w')
    for i in range(Experiment_result.shape[0]):
        f.write(str(i+1)+':'+str(Experiment_result[i,-2]) + '+/-'+str(Experiment_result[i,-1])+'\n')
    for i in range(Experiment_result.shape[1]-2):
        f.write('Experiment_num'+str(i) + '_OA:' + str(Experiment_result[0, i]) +'\n')
    f.close()



    print('Results Saving Finished!')


if __name__ == '__main__':
    main()
