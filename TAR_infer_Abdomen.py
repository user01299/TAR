import glob
import os, losses, utils3, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import SimpleITK as sitk
import torch
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion, distance_transform_edt
from torchvision import transforms
from scipy.ndimage import _ni_support
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted
from models import RDP
from ZReg.ZModel2 import ZRegIXI
from Comparison_Methods.CGNet import CGNet
from Comparison_Methods.CorrelationMorph import CorrelationMorph
from Comparison_Methods.TransMatch import TransMatch
import random
from Unet3D.UnetModel import UNet3D
import torch.nn as nn
from Comparison_Methods.ModelT import ModeT
from Comparison_Methods.TransMorph import TransMorph
from Comparison_Methods.TransMorph import CONFIGS as CONFIGS_TM
from Comparison_Methods.RDN import RDN
from Comparison_Methods.PAN import PAN
from Comparison_Methods.voxelM import U_Network
from SAT_REG.model.SAT_REG import SAT_REG_model_stage1, SAT_REG_model_stage2
from SAT_REG.dataset.SATdataset import LPBA_SAT_BrainDatasetS2S, SAT_Abd_DatasetS2S
def count_elements_in_3d_array(array):

    element_count = {}


    for layer in array:
        for row in layer:
            for element in row:

                element_count[element] = element_count.get(element, 0) + 1


    sorted_keys = sorted(element_count.keys())


    for num in sorted_keys:
        print(f"数值 {num} 出现了 {element_count[num]} 次")

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True

same_seeds(24)

def hausdorff_distance(result, reference, voxelspacing=None, connectivity=1, percentage=None):
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    if percentage is None:
        distance = max(hd1.max(), hd2.max())
    elif isinstance(percentage, (int, float)):
        distance = np.percentile(np.hstack((hd1, hd2)), percentage)
    else:
        raise ValueError
    return distance

def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype('bool'))
    reference = np.atleast_1d(reference.astype('bool'))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def main():
    config = {
        "test_data_dir": '../Unet3D/LPBA40/test/',
        "batch_size": 1,
        "num_workers": 1,
        "n_features": 1,
        "n_outputs": 14,
        "activation": "softmax",
        "base_width": 32,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "save_dir": '../Unet3D/best_models/abdomen/Channel32',
        #"save_dir": 'para/SATReg_LPBA_UnetLabel_unet_stage3_lr_0.0001_54r',
    }
    unetmodel = UNet3D(
        n_features=config["n_features"],
        n_outputs=config["n_outputs"],
        activation=config["activation"],
        base_width=config["base_width"],
        encoder_blocks=[2, 2, 2, 2],
        decoder_mirrors_encoder=True,
        downsampling_stride=2,
        interpolation_mode="trilinear"
    ).to(config["device"])


    def get_val_loss_from_filename(filename):
        loss_str = filename.split("valLoss_")[-1].split(".pth")[0]
        return float(loss_str)
    model_files = glob.glob(os.path.join(config["save_dir"], "epoch_*.pth"))
    if len(model_files) == 0:
        raise FileNotFoundError(f"模型目录 {config['save_dir']} 下未找到模型文件！")
    best_model_path = min(model_files, key=get_val_loss_from_filename)
    print(f"\n加载最佳模型：{os.path.basename(best_model_path)}")

    unetmodel.load_state_dict(torch.load(best_model_path, map_location=config["device"]))
    unetmodel.eval()


    # model_folder0 = 'SATReg_NCCpre_abdomen_UnetLabel_unet_stage3_lr_0.0001_54r/'
    # model_dir0 = 'para/' + model_folder0
    # model_filename0 = "dsc0.692.pth.tar"
    # best_stage0_para = torch.load(
    #     os.path.join(model_dir0, model_filename0),
    #     map_location="cuda:1"
    # )["state_dict"]
    # unetmodel.load_state_dict(best_stage0_para)

    # model_folder1 = 'SATReg_LPBA_UnetLabel_stage1_lr_0.0001_54r/'
    # model_folder2 = 'SATReg_LPBA_UnetLabel_stage2_lr_0.0001_54r/'

    model_folder1 = 'SATReg_AbdomenCT_unetLabel32_Model1_DiffPanPro_4channel_lr_0.0001_54r/'
    model_folder2 = 'SATReg_AbdomenCT_unetLabel32_Model1_DiffPanPro_Model2_PAN_lr_0.0001_54r/'




    model_dir1 = 'para/' + model_folder1
    model_filename1 = "dsc0.925.pth.tar"
    best_stage1_para = torch.load(
        os.path.join(model_dir1, model_filename1),
        map_location="cuda:1"
    )["state_dict"]


    model_dir2 = 'para/' + model_folder2
    model_filename2 = "dsc0.540.pth.tar"
    best_stage2_para = torch.load(
        os.path.join(model_dir2, model_filename2),
        map_location="cuda:1"
    )["state_dict"]

    #val_dir = '/media/user/04fa753c-89eb-47a9-9197-cd587b00f13b/liuyu/ZWH/RDP-main/Train_dataset/LPBA40/test'
    # test_dir = './dataset/LPBA/test/'
    test_dir = './dataset/OutAbdomenCTCT_cliped/test/'


    model_idx = -1


    csv_name = model_folder1[:-1]
    dict = utils3.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/' + csv_name + '.csv'):
        os.remove('Quantitative_Results/' + csv_name + '.csv')

    csv_writter(model_folder1[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line + ',' + 'non_jec', 'Quantitative_Results/' + csv_name)

    img_size = (160, 160, 192)
    #model = RDP(img_size, channels=16)

    model1 = SAT_REG_model_stage1().cuda()
    model1.load_state_dict(best_stage1_para)
    model2 = SAT_REG_model_stage2().cuda()
    # model2 = U_Network(3, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16]).cuda()
    #model2 = PAN(img_size).cuda()
    model2.load_state_dict(best_stage2_para)

    GPU_iden = 1
    # ********************************************************************8
    # torch.cuda.set_device(GPU_iden)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = ModeT(img_size, head_dim=head_dim, num_heads=num_heads, scale=1)

    # model = TransMorph(config)
    # model = U_Network(3, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16])
    # model = ModeT(inshape=img_size)
    lr = 0.0001
    # model_folder_comp = 'VoxlMorph_Abdomen_lr_{}_54r/'.format(lr)
    # model_dir_comp = 'para/' + model_folder_comp
    # model = U_Network(3, [16, 32, 32, 32], [32, 32, 32, 32, 32, 16, 16])
    # best_model_comp = \
    # torch.load(model_dir_comp + natsorted(os.listdir(model_dir_comp))[model_idx], map_location='cuda:0')[
    #     'state_dict']
    # print('Best model: {}'.format(natsorted(os.listdir(model_dir_comp))[model_idx]))
    # model.load_state_dict(best_model_comp)
    # model.cuda()


    reg_model = utils3.register_model(img_size, 'nearest')
    reg_model_img = utils3.register_model(img_size, 'bilinear')
    reg_model.cuda()
    image_composed = transforms.Compose([trans.NumpyType((np.float32, np.float32))])

    seg_composed = transforms.Compose([trans.SAT_Seg_norm_Abd(),
                                       trans.NumpyType((np.int16, np.int16))])

    #single modal
    #test_set = datasets.OASISBrainInferDatasetS2S(glob.glob(val_dir + '*.pkl'), transforms=test_composed)
    #IXIData
    # val_set = LPBA_SAT_BrainDatasetS2S(glob.glob(test_dir + '*.pkl'), transforms1=image_composed,
    #                                    transforms2=seg_composed)
    val_set = SAT_Abd_DatasetS2S(glob.glob(test_dir + '*.pkl'), transforms1=image_composed,
                                       transforms2=seg_composed)

    test_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)
    # *******************DATASET*******************************
    #Multi Modal
    #test_set = datasets.AbdomenMRCTInferDatasetS2S(glob.glob(val_dir_mr + '*.pkl'),glob.glob(val_dir_ct + '*.pkl'), transforms=test_composed)

    softeval_dsc_unet = utils.AverageMeter()
    eval_dsc_def = AverageMeter()
    eval_dsc = utils3.AverageMeter()
    eval_hd = utils3.AverageMeter()
    eval_asd = utils3.AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_Hos_det = AverageMeter()
    times = []
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            # model1.eval()
            # model2.eval()
            data = [t.cuda() for t in data]
            move_mask = data[0]
            fix_mask = data[1]
            move_label = data[2]
            fix_label = data[3]
            move = data[4]
            fix = data[5]  ## y = fix
            #________________________________________________________
            # 可以用于判断标签数量
            # xsee= fix_label.detach().cpu().numpy()[0, 0, ...]
            # count_elements_in_3d_array(xsee)
            with torch.no_grad():
                move_pred_prob = unetmodel(move)  # 概率图
                fix_pred_prob = unetmodel(fix)

                move_pred_class = torch.argmax(move_pred_prob, dim=1, keepdim=True)
                fix_pred_class = torch.argmax(fix_pred_prob, dim=1, keepdim=True)
                move_pred_maxprob = torch.max(move_pred_prob, dim=1, keepdim=True)[0]
                fix_pred_maxprob = torch.max(fix_pred_prob, dim=1, keepdim=True)[0]

                #
            mask_reg_out, mask_flow, move_masked_image, fix_masked_image, input_warpped_model2,vec = model1(
                move_pred_class, move_pred_maxprob, move,
                fix_pred_class, fix_pred_maxprob, fix,
                prob_threshold=0.5
            )
            mask_warred_move = input_warpped_model2
            stage2_fix = fix

            output, flow = model2(mask_warred_move, stage2_fix)

            # def_out = reg_model([move_label.float(), mask_flow])
            # def_out = reg_model([def_out.float(), flow])

            def_out1 = reg_model([move_label.cuda().float(), mask_flow.cuda()])
            def_out = reg_model([def_out1.cuda().float(), flow.cuda()])
            img_out1 = reg_model_img([move.cuda().float(), mask_flow.cuda()])
            img_out = reg_model_img([img_out1.cuda().float(), flow.cuda()])
            s
            tar = move.detach().cpu().numpy()[0, 0, :, :, :]
            #

            dsc, hd, asd = utils.metric_val_VOI_Abd(def_out.long(), fix_label.long())
            dsc_raw = utils.dice_raw_VOI_Abd(fix_label.long(), move_label.long())
            eval_dsc.update(dsc.item(), fix.size(0))
            eval_hd.update(hd.item(), fix.size(0))
            eval_asd.update(asd.item(), fix.size(0))


            jac_det = utils3.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), fix.size(0))
            stdy_idx += 1
            print('pair:{} -- dsc {}'.format(stdy_idx, eval_dsc.avg))
            print('hd {} '.format(eval_hd.avg))
            print('asd {}'.format(eval_asd.avg))
            print(' jacobian: {}'.format(eval_det.avg))


            # dsc_trans = utils.dice_val_VOI(def_out.long(), fix_seg.long())
            # dsc_raw = utils.dice_val_VOI(fix_seg.long(), move_seg.long())  #dice_val_VOI用于计算Dice
            # #
            # def_out_3d = torch.squeeze(def_out)
            # def_out_3d = def_out_3d.cpu().numpy()
            # fix_seg_3d = torch.squeeze(fix_seg)
            # fix_seg_3d = fix_seg_3d.cpu().numpy()
            # Hos = hausdorff_distance(def_out_3d, fix_seg_3d)

            # print('pair:{} Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(stdy_idx,dsc_trans.item(),dsc_raw.item()))
            # print('pair:{} hos: {}'.format(stdy_idx, Hos))
            # print('pair:{} jacobian: {}'.format(stdy_idx, eval_det.avg))
            #
            # eval_Hos_det.update(Hos.item(), fix.size(0))
            # eval_dsc_def.update(dsc_trans.item(), fix.size(0))
            eval_dsc_raw.update(dsc_raw.item(), fix.size(0))
            # stdy_idx += 1
        # print(' Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(
        #                                                                             eval_dsc_def.avg,
        #                                                                             eval_dsc_def.std,
        #                                                                             eval_dsc_raw.avg,
        #                                                                             eval_dsc_raw.std))
        # print('deformed jacobian det: {}, std: {}'.format(eval_det.avg, eval_det.std))
        # print('Hausduff 95 det: {}, std: {}'.format(eval_Hos_det.avg, eval_Hos_det.std))

        print(times)

        print('Sum---dsc{:.4f}dsc_std{:.4f}'.format(eval_dsc.avg, eval_dsc.std))
        print('Sum--dsc{:.4f}dsc_std{:.4f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))
        print('hd{:.4f}hd_std{:.4f}'.format(eval_hd.avg, eval_hd.std))
        print('asd{:.4f}asd_std{:.4f}'.format(eval_asd.avg, eval_asd.std))
        print('deformed jacobian det: {}, std: {}'.format(eval_det.avg, eval_det.std))


if __name__ == '__main__':

    '''
    GPU configuration
    '''
    GPU_iden = 1
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()




