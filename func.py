import torch
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
import cv2
import datetime
class load():
    # load dataset(indian_pines & pavia_univ.)
    def load_data(self,flag='indian'):
        if flag == 'indian':
            Ind_pines_dict = scio.loadmat('D:/HSI_data/Indian_pines_corrected.mat')
            Ind_pines_gt_dict = scio.loadmat('D:/HSI_data/Indian_pines_gt.mat')

            print(Ind_pines_dict['indian_pines_corrected'].shape)
            print(Ind_pines_gt_dict['indian_pines_gt'].shape)

            # remove the water absorption bands

            #no_absorption = list(set(np.arange(0, 103)) | set(np.arange(108, 149)) | set(np.arange(163, 219)))

            #original = Ind_pines_dict['indian_pines_gt'][:, :, no_absorption].reshape(145 * 145, 200)
            original = Ind_pines_dict['indian_pines_corrected'].reshape(145 * 145, 200)

            print(original.shape)
            print('Remove wate absorption bands successfully!')

            gt = Ind_pines_gt_dict['indian_pines_gt'].reshape(145 * 145, 1)

            r = Ind_pines_dict['indian_pines_corrected'].shape[0]
            c = Ind_pines_dict['indian_pines_corrected'].shape[1]
            categories = 17
        if flag == 'paviaC':
            pav_univ_dict = scio.loadmat('D:/HSI_data/Pavia.mat')
            pav_univ_gt_dict = scio.loadmat('D:/HSI_data/Pavia_gt.mat')

            print(pav_univ_dict['pavia'].shape)
            print(pav_univ_gt_dict['pavia_gt'].shape)

            #original = pav_univ_dict['pavia'].reshape(1096 * 715, 102)
            #gt = pav_univ_gt_dict['pavia_gt'].reshape(1096 * 715, 1)

            original = pav_univ_dict['pavia']
            gt = pav_univ_gt_dict['pavia_gt']

            original=original[:, -492:, :]
            gt=gt[:,-492:]
            print(original.shape)
            print(gt.shape)
            original=original.reshape(1096 * 492, 102)
            gt=gt.reshape(1096 * 492, 1)
            r = 1096
            c = 492
            categories = 10

        if flag == 'pavia':
            pav_univ_dict = scio.loadmat('D:/HSI_data/PaviaU.mat')
            pav_univ_gt_dict = scio.loadmat('D:/HSI_data/PaviaU_gt.mat')

            print(pav_univ_dict['paviaU'].shape)
            print(pav_univ_gt_dict['paviaU_gt'].shape)

            original = pav_univ_dict['paviaU'].reshape(610 * 340, 103)
            gt = pav_univ_gt_dict['paviaU_gt'].reshape(610 * 340, 1)

            r = pav_univ_dict['paviaU'].shape[0]
            c = pav_univ_dict['paviaU'].shape[1]
            categories = 10

        if flag == 'houston':
            houst_dict = scio.loadmat('D:/HSI_data/DFC2013_Houston.mat')
            houst_gt_dict = scio.loadmat('D:/HSI_data/DFC2013_Houston_gt.mat')

            print(houst_dict['DFC2013_Houston'].shape)
            print(houst_gt_dict['DFC2013_Houston_gt'].shape)

            original = houst_dict['DFC2013_Houston'].reshape(349 * 1905, 144)
            gt = houst_gt_dict['DFC2013_Houston_gt'].reshape(349 * 1905, 1)

            r = houst_dict['DFC2013_Houston'].shape[0]
            c = houst_dict['DFC2013_Houston'].shape[1]
            categories = 16

        if flag == 'salina':
            salinas_dict = scio.loadmat('./HSI_data/Salinas_corrected.mat')
            salinas_gt_dict = scio.loadmat('./HSI_data/Salinas_gt.mat')

            print(salinas_dict['salinas_corrected'].shape)
            print(salinas_gt_dict['salinas_gt'].shape)

            original = salinas_dict['salinas_corrected'].reshape(512 * 217, 204)
            gt = salinas_gt_dict['salinas_gt'].reshape(512 * 217, 1)

            r = salinas_dict['salinas_corrected'].shape[0]
            c = salinas_dict['salinas_corrected'].shape[1]
            categories = 17
        if flag == 'ksc':
            salinas_dict = scio.loadmat('D:/HSI_data/KSC.mat')
            salinas_gt_dict = scio.loadmat('D:/HSI_data/KSC_gt.mat')

            print(salinas_dict['KSC'].shape)
            print(salinas_gt_dict['KSC_gt'].shape)

            original = salinas_dict['KSC'].reshape(512 * 614, 176)
            gt = salinas_gt_dict['KSC_gt'].reshape(512 * 614, 1)

            r = salinas_dict['KSC'].shape[0]
            c = salinas_dict['KSC'].shape[1]
            categories = 14

        rows = np.arange(gt.shape[0])  # start from 0
        # ID(row number), data, class number
        All_data = np.c_[rows, original, gt]

        # Removing background and obtain all labeled data
        labeled_data = All_data[All_data[:, -1] != 0, :]
        rows_num = labeled_data[:, 0]  # All ID of labeled  data

        return All_data, labeled_data, rows_num, categories, r, c, flag


class product():
    def __init__(self, c, flag, All_data):
        self.c=c
        self.flag = flag
        self.All_data = All_data
    # product the training and testing pixel ID
    def generation_num(self, labeled_data, rows_num,ITER):

        train_num = []

        for i in np.unique(labeled_data[:, -1]):
            temp = labeled_data[labeled_data[:, -1] == i, :]
            temp_num = temp[:, 0]  # all ID of a special class
            #print(i, temp_num.shape[0])
            np.random.seed()
            np.random.shuffle(temp_num)  # random sequence

            if self.flag == 'indian':    #80num

                #train_num.append(temp_num[0:10])

                if i == 1:
                    train_num.append(temp_num[0:26])
                elif i == 7:
                    train_num.append(temp_num[0:16])
                elif i == 9:
                    train_num.append(temp_num[0:11])
                elif i == 16:
                    train_num.append(temp_num[0:60])
                else:
                    train_num.append(temp_num[0:80])


            '''
            if self.flag == 'indian':  # 200num
                
                train_num.append(temp_num[0:5])
            
                if i == 1:
                   train_num.append(temp_num[0:26])
                elif i == 7:
                    train_num.append(temp_num[0:16])
                elif i == 9:
                    train_num.append(temp_num[0:11])
                # elif i == 16:
                #    train_num.append(temp_num[0:75])
                else:
                    train_num.append(temp_num[0:50])
            '''


            if self.flag == 'pavia' or self.flag=='houston' or self.flag=='salina'or self.flag=='paviaC':
                train_num.append(temp_num[0:80])
            if self.flag == 'ksc':
                if i == 5:
                    train_num.append(temp_num[0:40])
                elif i == 7:
                    train_num.append(temp_num[0:40])
                else:
                    train_num.append(temp_num[0:80])
        #             else:
        #                 train_num.append(temp_num[0:int(temp.shape[0]*0.class_num.py)])

        trn_num = [x for j in train_num for x in j]  # merge
        np.random.seed(ITER+123456)
        np.random.shuffle(trn_num)
        val_num = trn_num[int(len(trn_num)*0.8):]
        #tes_num = list(set(rows_num) - set(trn_num))
        tes_num=rows_num
        pre_num = list(set(range(0, self.All_data.shape[0])) - set(trn_num))
        print('number of training sample', int(len(trn_num)))
        return rows_num, trn_num, val_num, tes_num, pre_num


    def production_label(self, num, y_map, split='Trn'):

        num = np.array(num)
        idx_2d = np.zeros([num.shape[0], 2]).astype(int)
        idx_2d[:, 0] = num // self.c
        idx_2d[:, 1] = num % self.c

        label_map = np.zeros(y_map.shape)
        for i in range(num.shape[0]):
            label_map[idx_2d[i,0],idx_2d[i,1]] = self.All_data[int(num[i]),-1]

        print('{} label map preparation Finished!'.format(split))
        return label_map

def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - class_num.py.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]#output上分对的类别
    # https://github.com/pytorch/pytorch/issues/1382
    area_intersection = torch.histc(intersection.float().cpu(), bins=K, min=0, max=K-1)#output上分对的类别中每类的个数
    area_output = torch.histc(output.float().cpu(), bins=K, min=0, max=K-1)#output每类的个数
    area_target = torch.histc(target.float().cpu(), bins=K, min=0, max=K-1)#target每类的个数
    area_union = area_output + area_target - area_intersection
    return area_intersection.cuda(), area_union.cuda(), area_target.cuda()



def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        '''
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([100, 255, 100])/255.
        if item == 3:
            y[index] = np.array([0,0,255])/255.
        if item == 4:
            y[index] = np.array([255, 255, 0])/255.
        if item == 5:
            y[index] = np.array([255, 0, 255])/255.
        if item == 6:
            y[index] = np.array([255, 100, 100])/255.
        if item == 7:
            y[index] = np.array([150, 75, 255])/255.
        if item == 8:
            y[index] = np.array([150, 75, 75])/255.
        if item == 9:
            y[index] = np.array([100, 100, 255])/255.
        if item == 10:
            y[index] = np.array([0, 200, 200])/255.
        if item == 11:
            y[index] = np.array([0, 100, 100])/255.
        if item == 12:
            y[index] = np.array([100, 0, 100])/255.
        if item == 13:
            y[index] = np.array([128, 128, 0])/255.
        if item == 14:
            y[index] = np.array([200, 100, 0])/255.
        if item == 15:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 16:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 17:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 18:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 19:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 20:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 21:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 22:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 23:
            y[index] = np.array([0, 255, 0]) / 255.
        '''
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255,182,193]) / 255.
        if item == 2:
            y[index] = np.array([60,179,113]) / 255.
        if item == 3:
            y[index] = np.array([255,165,0]) / 255.
        if item == 4:
            y[index] = np.array([65,105,225]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 0]) / 255.
        if item == 6:
            y[index] = np.array([148,0,211]) / 255.
        if item == 7:
            y[index] = np.array([139,69,19]) / 255.
        if item == 8:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 9:
            y[index] = np.array([0,255,255])/255.
        if item == 10:
            y[index] = np.array([128, 128, 0])/255.
        if item == 11:
            y[index] = np.array([255,255,0])/255.
        if item == 12:
            y[index] = np.array([121,255,49])/255.
        if item == 13:
            y[index] = np.array([255,49,183])/255.
        if item == 14:
            y[index] = np.array([112, 192, 188])/255.
        if item == 15:
            y[index] = np.array([183,121,121])/255.
        if item == 16:
            y[index] = np.array([13,0,100])/255.

    return y


def generate_png(gt_hsi,pred_test,flag,h,w,total_indices):


    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)

    for i in range(len(pred_test)):
        pred_test[i] = pred_test[i] + 1

    for i in range(len(gt)):
        if gt[i] == 255.0:
            gt[i] = 0.0
        else:
            gt[i] +=1.0

    #for i in range(gt.shape[0]):  #画局部图
    #    if gt[i] == 0: pred_test[i]=0

    y_list = list_to_colormap(pred_test)
    y_gt = list_to_colormap(gt)


    y_re = np.reshape(y_list, (h, w, 3))
    gt_re = np.reshape(y_gt, (h, w, 3))

    day = datetime.datetime.now()
    day_str = day.strftime('%m_%d_%H_%M')

    path = './maps/'
    classification_map(y_re, gt_re, 600,
                       path + '_' + 'Time_'+str(day_str)+'_'+str(flag)+'.eps')
    #classification_map(gt_re, gt_re, 600,
    #                   path + 'Time_gt'+str(day_str)+'_'+str(flag)+'.eps')
    print('------Get classification maps successful-------')