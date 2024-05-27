import os
import torch
import numpy as np

from tqdm import tqdm
from torchvision.utils import save_image
from utils.option import parser_args
from utils.utils import get_test_loaders
from utils.logger import Logger
from metric.psnr_ssim import SSIMs_PSNRs
from pytorch_fid import fid_score


dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

opt = parser_args()
test_loader = get_test_loaders(opt)

path = ' '   # the path of the model
G_A2B = torch.load(path)
G_B2A = torch.load(path)
G_A2B.eval()
G_B2A.eval()
tsl = tqdm(test_loader)

os.makedirs("save/vis/%s/%s/test/A" % (opt.dataset_name, opt.model_name), exist_ok=True)
os.makedirs("save/vis/%s/%s/test/B" % (opt.dataset_name, opt.model_name), exist_ok=True)

test_logger_path = os.path.join("save/log/%s/%s" % (opt.dataset_name, opt.model_name), 'test_log.txt')
test_logger = Logger(test_logger_path)

with torch.no_grad():
    for _, [index, batch_img1, batch_img2] in enumerate(tsl):
        val_batch_iter = val_batch_iter + opt.batch_size
        batch_img1 = batch_img1.float().to(dev)
        batch_img2 = batch_img2.float().to(dev)
        real_A = torch.autograd.Variable(batch_img1).cuda()
        real_B = torch.autograd.Variable(batch_img2).cuda()
        fake_B = G_A2B(real_A)
        fake_A = G_B2A(real_B)

        save_image(fake_A, f"save/vis/%s/%s/test/A/{index}.png" % (opt.dataset_name, opt.model_name),
                   normalize=True)
        save_image(fake_B, f"save/vis/%s/%s/test/B/{index}.png" % (opt.dataset_name, opt.model_name),
                   normalize=True)

    real_A_dataset_path = os.path.join(opt.dataset_dir, 'test/A')
    fake_A_dataset_path = 'save/vis/%s/%s/test/A' % (opt.dataset_name, opt.model_name)
    fid_A_paths = [real_A_dataset_path, fake_A_dataset_path]
    fid_A_score = fid_score.calculate_fid_given_paths(fid_A_paths, batch_size=opt.fid_batchsize, device=dev,
                                                      dims=opt.fid_dims)

    real_B_dataset_path = os.path.join(opt.dataset_dir, 'test/B')
    fake_B_dataset_path = 'save/vis/%s/%s/test/B' % (opt.dataset_name, opt.model_name)
    fid_B_paths = [real_B_dataset_path, fake_B_dataset_path]
    fid_B_score = fid_score.calculate_fid_given_paths(fid_B_paths, batch_size=opt.fid_batchsize, device=dev,
                                                      dims=opt.fid_dims)

    message_val_fid_score = 'fid_A_score: %.5f, fid_B_score: %.5f' % \
                            (fid_A_score, fid_B_score)
    test_logger.write(message_val_fid_score)
    test_logger.write('\n')

    ssim_measures, psnr_measures = SSIMs_PSNRs(real_B_dataset_path, fake_B_dataset_path)
    ssim_len = len(ssim_measures)
    ssim_mean = np.mean(ssim_measures)
    ssim_std = np.std(ssim_measures)
    psnr_len = len(psnr_measures)
    psnr_mean = np.mean(psnr_measures)
    psnr_std = np.std(psnr_measures)

    message_val_ssim_psnr = 'ssim_len: {0}, ssim_mean: {1}, ssim_std: {2}, ' \
                            'psnr_len: {3}, psnr_mean: {4}, psnr_std: {5},' \
        .format(ssim_len, ssim_mean, ssim_std, psnr_len, psnr_mean, psnr_std)
    test_logger.write(message_val_ssim_psnr)
    test_logger.write('\n')
