import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import random
import logging
import itertools

import numpy as np
import torch

from torchvision.utils import save_image
from tqdm import tqdm
from model.model_utils import weights_init_normal, weights_init
from utils.logger import Logger
from utils.option import parser_args
from utils.utils import get_loaders, LambdaLR, ReplayBuffer, sample_images_in_training
from utils.utils import load_A2BGAN_generator, load_B2AGAN_generator, load_A_GAN_discriminator, load_B_GAN_discriminator
from pytorch_fid import fid_score
from metric.psnr_ssim import SSIMs_PSNRs
from torch.utils.tensorboard import SummaryWriter


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    opt = parser_args()
    print('start')
    os.makedirs("save/checkpoint/%s/%s" % (opt.dataset_name, opt.model_name), exist_ok=True)
    os.makedirs("save/log/%s/%s" % (opt.dataset_name, opt.model_name), exist_ok=True)
    os.makedirs("save/vis/%s/%s/train" % (opt.dataset_name, opt.model_name), exist_ok=True)
    # os.makedirs("save/vis/%s/%s/val/A" % (opt.dataset_name, opt.model_name), exist_ok=True)
    # os.makedirs("save/vis/%s/%s/val/B" % (opt.dataset_name, opt.model_name), exist_ok=True)

    writer = SummaryWriter(log_dir=' ')

    logging.basicConfig(level=logging.INFO)

    if torch.cuda.is_available():
        print('num of GPU:', torch.cuda.device_count())
        dev = torch.device('cuda')
        opt.cuda = True
    else:
        dev = torch.device('cpu')
        opt.cuda = False

    logging.info('GPU AVAILABLE? ' + str(torch.cuda.is_available()))

    seed_torch(opt.seed)
    train_data_loader, val_data_loader = get_loaders(opt)

    logging.info('LOADING Model')
    G_A2B = load_A2BGAN_generator(opt)
    G_B2A = load_B2AGAN_generator(opt)
    D_A = load_A_GAN_discriminator(opt)
    D_B = load_B_GAN_discriminator(opt)

    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    if opt.cuda:
        criterion_GAN.cuda()
        criterion_cycle.cuda()
        criterion_identity.cuda()

    optimizer_G = torch.optim.Adam(
        itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=opt.lr_gen, betas=(opt.b1, opt.b2)
    )
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr_dis, betas=(opt.b1, opt.b2))
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr_dis, betas=(opt.b1, opt.b2))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
        optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
        optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.start_epoch, opt.decay_epoch).step)

    G_A2B.apply(weights_init_normal)
    G_B2A.apply(weights_init_normal)
    D_A.apply(weights_init('gaussian'))
    D_B.apply(weights_init('gaussian'))

    train_logger_path = os.path.join("save/log/%s/%s" % (opt.dataset_name, opt.model_name), 'train_log.txt')
    train_logger = Logger(train_logger_path)
    val_logger_path = os.path.join("save/log/%s/%s" % (opt.dataset_name, opt.model_name), 'val_log.txt')
    val_logger = Logger(val_logger_path)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    best_fid_A_score = 500.0
    best_fid_B_score = 500.0
    psnr = 0.0
    ssim = 0.0
    best_epoch_id = 0
    train_batch_iter = 0
    val_batch_iter = 0
    logging.info('STARTING training')

    for epoch in range(opt.start_epoch, opt.n_epochs):
        os.makedirs("save/vis/%s/%s/val/%d/A" % (opt.dataset_name, opt.model_name, epoch), exist_ok=True)
        os.makedirs("save/vis/%s/%s/val/%d/B" % (opt.dataset_name, opt.model_name, epoch), exist_ok=True)

        tdl = tqdm(train_data_loader)
        train_logger.write('Begin training...')
        train_logger.write('\n')

        loss_gen_adv_a_print = []
        loss_gen_adv_b_print = []
        loss_D_A_print = []
        loss_D_B_print = []
        loss_cycle_print = []
        loss_identity_print = []

        for i, [index, batch_img1, batch_img2] in enumerate(tdl):
            tdl.set_description(
                "epoch {} info ".format(epoch) + str(train_batch_iter) + " - " + str(train_batch_iter + opt.batch_size))
            train_batch_iter = train_batch_iter + opt.batch_size

            batch_img1 = batch_img1.float().to(dev)
            batch_img2 = batch_img2.float().to(dev)

            real_A = torch.autograd.Variable(batch_img1).cuda()
            real_B = torch.autograd.Variable(batch_img2).cuda()

            G_A2B.train()
            G_B2A.train()

            loss_id_A = criterion_identity(G_B2A(real_B), real_A)
            loss_id_B = criterion_identity(G_A2B(real_A), real_B)
            loss_identity = (loss_id_A + loss_id_B) / 2

            loss_identity_print.append(loss_identity.data.cpu().numpy())

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            loss_gen_adv_a = 0
            loss_gen_adv_b = 0
            D_B_outs = D_B(fake_B)
            for it, (out) in enumerate(D_B_outs):
                loss_gen_adv_b += torch.mean((out - 1) ** 2)
            D_A_outs = D_A(fake_A)
            for it, (out) in enumerate(D_A_outs):
                loss_gen_adv_a += torch.mean((out - 1) ** 2)

            recov_A = G_B2A(fake_B)
            loss_cycle_A = criterion_cycle(recov_A, real_A)
            recov_B = G_A2B(fake_A)
            loss_cycle_B = criterion_cycle(recov_B, real_B)
            loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

            loss_cycle_print.append(loss_cycle.data.cpu().numpy())

            loss_gen_adv_a_print.append(loss_gen_adv_a.data.cpu().numpy())
            loss_gen_adv_b_print.append(loss_gen_adv_b.data.cpu().numpy())

            loss_G = opt.lambda_gen * loss_gen_adv_a + opt.lambda_gen * loss_gen_adv_b + opt.lambda_cyc * loss_cycle + \
                     opt.lambda_id * loss_identity
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # Train Discriminator A

            fake_A_ = fake_A_buffer.push_and_pop(fake_A)

            D_A_outs0 = D_A(fake_A_)
            D_A_outs1 = D_A(real_A)
            loss_dis_a = 0
            for it, (outs0, outs1) in enumerate(zip(D_A_outs0, D_A_outs1)):
                loss_dis_a += torch.mean((outs0 - 0) ** 2) + torch.mean((outs1 - 1) ** 2)
            loss_D_A = opt.lambda_gen * loss_dis_a

            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()

            # Train Discriminator B

            fake_B_ = fake_B_buffer.push_and_pop(fake_B)

            D_B_outs0 = D_B(fake_B_)
            D_B_outs1 = D_B(real_B)
            loss_dis_b = 0
            for it, (outs0, outs1) in enumerate(zip(D_B_outs0, D_B_outs1)):
                loss_dis_b += torch.mean((outs0 - 0) ** 2) + torch.mean((outs1 - 1) ** 2)
            loss_D_B = opt.lambda_gen * loss_dis_b

            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            loss_D_A_print.append(loss_D_A.data.cpu().numpy())
            loss_D_B_print.append(loss_D_B.data.cpu().numpy())

            loss_D = (loss_D_A + loss_D_B) / 2

            train_len = len(train_data_loader)

            if np.mod(train_batch_iter, 500) == 1:
                message_train_loss = '[Epoch %d,%d], [Batch %d,%d], G_loss: %.5f, D_loss: %.5f, cycle_loss: %.5f, ' \
                                     'identity_loss: %.5f' % \
                                     (epoch, opt.n_epochs - 1, i, train_len, loss_G, loss_D, loss_cycle, loss_identity)
                train_logger.write('\n')
                train_logger.write(message_train_loss)

            if train_batch_iter % opt.sample_interval == 0:
                sample_images_in_training(opt, epoch, train_batch_iter, val_data_loader, G_A2B, G_B2A)

        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        loss_gen_adv_a_mean = np.mean(loss_gen_adv_a_print)
        loss_gen_adv_b_mean = np.mean(loss_gen_adv_b_print)
        loss_D_A_mean = np.mean(loss_D_A_print)
        loss_D_B_mean = np.mean(loss_D_B_print)
        loss_cycle_mean = np.mean(loss_cycle_print)
        loss_identity_mean = np.mean(loss_identity_print)

        writer.add_scalar(tag='loss_g_a_mean', scalar_value=loss_gen_adv_a_mean, global_step=epoch)
        writer.add_scalar(tag='loss_g_b_mean', scalar_value=loss_gen_adv_b_mean, global_step=epoch)
        writer.add_scalar(tag='loss_D_A_mean', scalar_value=loss_D_A_mean, global_step=epoch)
        writer.add_scalar(tag='loss_D_B_mean', scalar_value=loss_D_B_mean, global_step=epoch)
        writer.add_scalar(tag='loss_cycle_mean', scalar_value=loss_cycle_mean, global_step=epoch)
        writer.add_scalar(tag='loss_identity_mean', scalar_value=loss_identity_mean, global_step=epoch)

        '''
        Validation
        '''
        val_logger.write('Begin evaluation...')
        val_logger.write('\n')
        G_A2B.eval()
        G_B2A.eval()
        tvl = tqdm(val_data_loader, desc='\r')

        with torch.no_grad():
            for _, [index, batch_img1, batch_img2] in enumerate(tvl):
                tvl.set_description(
                    "epoch {} info ".format(epoch) + str(val_batch_iter) + " - " + str(val_batch_iter + opt.batch_size))
                val_batch_iter = val_batch_iter + opt.batch_size
                batch_img1 = batch_img1.float().to(dev)
                batch_img2 = batch_img2.float().to(dev)
                real_A = torch.autograd.Variable(batch_img1).cuda()
                real_B = torch.autograd.Variable(batch_img2).cuda()
                fake_B = G_A2B(real_A)
                fake_A = G_B2A(real_B)

                save_image(fake_A, f"save/vis/%s/%s/val/%d/A/{index}.png" % (opt.dataset_name, opt.model_name, epoch), normalize=True)
                save_image(fake_B, f"save/vis/%s/%s/val/%d/B/{index}.png" % (opt.dataset_name, opt.model_name, epoch), normalize=True)

            real_A_dataset_path = os.path.join(opt.dataset_dir, 'val/A')
            fake_A_dataset_path = 'save/vis/%s/%s/val/%d/A' % (opt.dataset_name, opt.model_name, epoch)
            fid_A_paths = [real_A_dataset_path, fake_A_dataset_path]
            fid_A_score = fid_score.calculate_fid_given_paths(fid_A_paths, batch_size=opt.fid_batchsize, device=dev,
                                                    dims=opt.fid_dims)

            real_B_dataset_path = os.path.join(opt.dataset_dir, 'val/B')
            fake_B_dataset_path = 'save/vis/%s/%s/val/%d/B' % (opt.dataset_name, opt.model_name, epoch)
            fid_B_paths = [real_B_dataset_path, fake_B_dataset_path]
            fid_B_score = fid_score.calculate_fid_given_paths(fid_B_paths, batch_size=opt.fid_batchsize, device=dev,
                                                    dims=opt.fid_dims)

            message_val_fid_score = '[Epoch %d,%d], fid_A_score: %.5f, fid_B_score: %.5f' % \
                                    (epoch, opt.n_epochs - 1, fid_A_score, fid_B_score)
            val_logger.write(message_val_fid_score)
            val_logger.write('\n')

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
            val_logger.write(message_val_ssim_psnr)
            val_logger.write('\n')

            if ssim_mean > ssim:
                ssim = ssim_mean
                val_logger.write('ssim: {}'.format(ssim))
                val_logger.write('\n')
            else:
                val_logger.write('ssim: {}'.format(ssim))
                val_logger.write('\n')

            if psnr_mean > psnr:
                psnr = psnr_mean
                val_logger.write('psnr: {}'.format(psnr))
                val_logger.write('\n')
            else:
                val_logger.write('psnr: {}'.format(psnr))
                val_logger.write('\n')

        torch.save({
            'epoch': epoch,
            'best_fid_B_score': best_fid_B_score,
            'best_epoch': best_epoch_id,
            'G_A2B_state_dict': G_A2B.state_dict(),
            'G_B2A_state_dict': G_B2A.state_dict(),
            'D_A_state_dict': D_A.state_dict(),
            'D_B_state_dict': D_B.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
            'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
            'scheduler_G_state_dict': lr_scheduler_G.state_dict(),
            'scheduler_D_A_state_dict': lr_scheduler_D_A.state_dict(),
            'scheduler_D_B_state_dict': lr_scheduler_D_B.state_dict(),
        }, os.path.join("save/checkpoint/%s/%s" % (opt.dataset_name, opt.model_name), 'last_ckpt.pt'))
        val_logger.write('Lastest model updated. fid_A_score=%.4f, fid_B_score=%.4f, Historical_best_fid_A_score=%.4f, '
                         'Historical_best_fid_B_score=%.4f (at epoch %d)' % (fid_A_score, fid_B_score, best_fid_A_score, best_fid_B_score, best_epoch_id))
        val_logger.write('\n')

        if fid_B_score < best_fid_B_score:
            best_fid_B_score = fid_B_score
            best_epoch_id = epoch
            torch.save({
                'epoch': epoch,
                'best_fid_A_score': best_fid_B_score,
                'best_epoch': best_epoch_id,
                'G_A2B_state_dict': G_A2B.state_dict(),
                'G_B2A_state_dict': G_B2A.state_dict(),
                'D_A_state_dict': D_A.state_dict(),
                'D_B_state_dict': D_B.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optim   izer_D_A_state_dict': optimizer_D_A.state_dict(),
                'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
                'scheduler_G_state_dict': lr_scheduler_G.state_dict(),
                'scheduler_D_A_state_dict': lr_scheduler_D_A.state_dict(),
                'scheduler_D_B_state_dict': lr_scheduler_D_B.state_dict(),
            }, os.path.join("save/checkpoint/%s/%s" % (opt.dataset_name, opt.model_name), 'best_ckpt.pt'))
            val_logger.write('*' * 7 + 'Best model updated!')
            val_logger.write('\n')

        train_logger.write('*' * 7 + 'An epoch finished.' + '*' * 7)
        train_logger.write('\n')
        val_logger.write('*' * 7 + 'An epoch finished.' + '*' * 7)
        val_logger.write('\n')


if __name__ == '__main__':
    main()
