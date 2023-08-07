import random
import torch
import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ts_generation import ts_generation
from Model import SRN
from sklearn import metrics
from typing import Dict
from utils import standard, checkFile, getSAM, getPSNR

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def printStatus(train_step, x, y, loss, loss1, loss2, lr):
    '''
    To show the information during training
    '''
    train_loss = loss.item()
    mapping_loss = loss1.item()
    energy_loss = loss2.item()
    train_psnr = getPSNR(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    train_sam = getSAM(x.cpu().detach().numpy(), y.cpu().detach().numpy())
    print('Iteration%s----LR:%s----psnr:%s--sam:%s-loss:%s-energy_loss:%s---mapping_loss:%s' % (
        train_step, lr, train_psnr, train_sam, train_loss, energy_loss, mapping_loss))


def calculate_loss(RX, X, F, t_tensor, t_proj, S, eta1):
    '''
    The loss function including the reconstruction loss, the sparse loss, and the CEM loss
    :param RX: The reconstruction result of HSI by subspace representation
    :param X: The input HSI
    :param F: The detection map of X
    :param t_tensor: The target cube tiled by the target spectrum t
    :param t_proj: The detection map of the target cube
    :param S: The abundances containing the background coefficients and target coefficients
    :param eta1: The regularization parameter controlling the degree of sparsity
    return: The total loss, the division of the CEM loss
    '''
    ## The reconstruction loss
    recon_loss = torch.mean(torch.abs(RX - X))

    ## The CEM loss
    weight = torch.sum(torch.square(X - t_tensor), dim=1)
    mapping_loss = torch.mean(torch.abs(t_proj - 1))
    energy_loss = torch.mean(weight * torch.square(F))
    CEM_loss = 0.1 * energy_loss + mapping_loss

    ## The sparse loss, i.e., the sparsity of abundances using the L1-norm
    spar_loss = torch.mean(torch.sum(torch.abs(S), dim=1))

    return recon_loss + CEM_loss + eta1 * spar_loss, energy_loss, mapping_loss


def data_preprocessing(modelConfig):
    ## loading data
    mat = sio.loadmat(modelConfig['dataset'])
    hs = mat['data']
    gt = mat['map']

    ## pre-processing
    hs[hs < 0] = 0
    hs = standard(hs)
    hs = np.float32(hs)
    H, W, C = hs.shape

    ## generate the target spectrum
    # select the pixel with the smallest Euclidean distance to other pixels as the target spectrum
    target_spectrum = ts_generation(hs, gt, 5)

    ## convert the hsi into the tensor form of pytorch
    hs_tensor = torch.from_numpy(hs)  # 100*100*189
    hs_tensor = torch.unsqueeze(hs_tensor, 0)  # 1*100*100*189
    hs_tensor = torch.permute(hs_tensor, (0, 3, 1, 2))  # 1*189*100*100

    ## convert the target spectrum into the atom form and tensor form
    t_tensor = torch.from_numpy(target_spectrum)  # 189*1
    t_tensor = torch.unsqueeze(t_tensor, -1)  # 189*1*1
    t_atom = torch.unsqueeze(t_tensor, -1)  # target subspace, 189*1*1*1
    t_tensor = torch.unsqueeze(t_tensor, 0)  # tensor form of target spectrum, 1*189*1*1
    # the target cube tiled by the target spectrum
    et_tensor = torch.tile(t_tensor, (1, 1, H, W))  # 1*189*100*100

    return gt, hs_tensor, t_atom, t_tensor, et_tensor


def train(modelConfig: Dict):
    ## fix the random seed for stable performance
    seed_torch(1)
    ## pre-processing
    _, hs_tensor, t_atom, t_tensor, et_tensor = data_preprocessing(modelConfig)
    ## Model
    model = SRN(hs_tensor.shape[1], modelConfig['m']).cuda()
    model.train()

    # defining the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=modelConfig['lr'], weight_decay=modelConfig['weight_decay'])
    dir_path = modelConfig['model_path'] + modelConfig['dataset']
    checkFile(dir_path)

    for i in range(1, modelConfig['iteration'] + 1):
        optimizer.zero_grad()
        predict_x, detection_map, S = model(hs_tensor.cuda(), t_atom.cuda())
        _, t_proj, _ = model(et_tensor.cuda(), t_atom.cuda())
        loss, energy_loss, mapping_loss = calculate_loss(predict_x, hs_tensor.cuda(), detection_map, t_tensor.cuda(),
                                                         t_proj,
                                                         S, modelConfig['eta1'])

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            printStatus(i, predict_x, hs_tensor.cuda(), loss, energy_loss, mapping_loss,
                        optimizer.param_groups[0]['lr'])

        if i % 1000 == 0:
            lr = optimizer.param_groups[0]['lr'] * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    ## save the model after the last iteration
    torch.save(model.state_dict(), dir_path + '/m_%s_eta1_%s' % (modelConfig['m'], modelConfig['eta1']))


def test(modelConfig: Dict):
    ## fix the random seed for stable performance
    seed_torch(1)

    ## pre-processing
    gt, hs_tensor, t_atom, t_tensor, et_tensor = data_preprocessing(modelConfig)

    ## Model
    model = SRN(hs_tensor.shape[1], modelConfig['m']).cuda()
    ## Loading the saved model parameters
    dir_path = modelConfig['model_path'] + modelConfig['dataset']
    model.load_state_dict(torch.load(dir_path + '/m_%s_eta1_%s' % (modelConfig['m'], modelConfig['eta1'])))
    model.eval()

    predict_x, detection_map, S = model(hs_tensor.cuda(), t_atom.cuda())

    detection_map = detection_map.cpu().detach().numpy()
    detection_map = np.squeeze(detection_map, axis=0)

    detection_map = standard(detection_map)
    detection_map = np.clip(detection_map, 0, 1)

    y_l = np.reshape(gt, [-1, 1], order='F')
    y_p = np.reshape(detection_map, [-1, 1], order='F')

    ## calculate the AUC value
    fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]
    auc1 = round(metrics.auc(fpr, tpr), modelConfig['epision'])
    auc2 = round(metrics.auc(threshold, fpr), modelConfig['epision'])
    auc3 = round(metrics.auc(threshold, tpr), modelConfig['epision'])
    auc4 = round(auc1 + auc3 - auc2, modelConfig['epision'])
    auc5 = round(auc3 / auc2, modelConfig['epision'])
    print('{:.{precision}f}'.format(auc1, precision=modelConfig['epision']))
    print('{:.{precision}f}'.format(auc2, precision=modelConfig['epision']))
    print('{:.{precision}f}'.format(auc3, precision=modelConfig['epision']))
    print('{:.{precision}f}'.format(auc4, precision=modelConfig['epision']))
    print('{:.{precision}f}'.format(auc5, precision=modelConfig['epision']))

    plt.imshow(detection_map)
    plt.show()


def test_auc(modelConfig: Dict):
    ## fix the random seed for stable performance
    seed_torch(1)

    ## pre-processing
    gt, hs_tensor, t_atom, t_tensor, et_tensor = data_preprocessing(modelConfig)

    ## Model
    model = SRN(hs_tensor.shape[1], modelConfig['m']).cuda()
    ## Loading the saved model parameters
    dir_path = modelConfig['model_path'] + modelConfig['dataset']
    model.load_state_dict(torch.load(dir_path + '/m_%s_eta1_%s' % (modelConfig['m'], modelConfig['eta1'])))
    model.eval()

    predict_x, detection_map, S = model(hs_tensor.cuda(), t_atom.cuda())

    detection_map = detection_map.cpu().detach().numpy()
    detection_map = np.squeeze(detection_map, axis=0)

    detection_map = standard(detection_map)
    detection_map = np.clip(detection_map, 0, 1)

    y_l = np.reshape(gt, [-1, 1], order='F')
    y_p = np.reshape(detection_map, [-1, 1], order='F')

    ## calculate the AUC value
    fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
    fpr = fpr[1:]
    tpr = tpr[1:]
    threshold = threshold[1:]
    auc = round(metrics.auc(fpr, tpr), modelConfig['epision'])
    return auc


def parameter_selection(modelConfig: Dict):
    '''
    To select the optimal m and eta1
    '''
    for m in [1, 5, 10, 15, 20, 25, 30]:
        for eta1 in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]:
            modelConfig['m'] = m
            modelConfig['eta1'] = eta1
            train(modelConfig)
    max_auc = 0
    opt_m = 0
    opt_eta1 = 0
    for m in [1, 5, 10, 15, 20, 25, 30]:
        for eta1 in [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]:
            modelConfig['m'] = m
            modelConfig['eta1'] = eta1
            try:
                auc = test_auc(modelConfig)
            except:
                auc = 0.5
            if auc > max_auc:
                max_auc = auc
                opt_m = m
                opt_eta1 = eta1
    print('The selected m and eta1 are %s and %s, auc=%.4f' % (opt_m, opt_eta1, max_auc))
    with open('m_eta1.txt', 'at') as f:
        f.write('%s: m=%s, eta1=%s, auc=%.4f' % (modelConfig['dataset'], opt_m, opt_eta1, max_auc))
    modelConfig['m'] = opt_m
    modelConfig['eta1'] = opt_eta1
    test(modelConfig)
