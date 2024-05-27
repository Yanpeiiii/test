import random
import torch
import torch.nn as nn
import logging

from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from data.dataloader import full_path_loader, full_test_loader, Loader
from model.utgan import DTGAN, Discriminator
from model.discriminator import MsImageDiscriminator


class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):                                              # return    1-max(0, epoch - 30) / (50 - 30)
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def get_loaders(opt):
    logging.info('STARTING Dataset Creation')

    train_full_load, val_full_load = full_path_loader(opt.dataset_dir)

    train_dataset = Loader(train_full_load, aug=True)
    val_dataset = Loader(val_full_load, aug=False)

    logging.info('STARTING Dataloading')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batch_size,
                                                   shuffle=True,
                                                   num_workers=opt.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=opt.batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return train_loader, val_loader


def get_test_loaders(opt, batch_size=None):

    if not batch_size:
        batch_size = opt.batch_size

    logging.info('STARTING Dataset Creation')

    test_full_load = full_test_loader(opt.dataset_dir)

    test_dataset = Loader(test_full_load, aug=False)

    logging.info('STARTING Dataloading')


    test_loader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=opt.num_workers)
    return test_loader


def load_A2BGAN_generator(opt):
    model = DTGAN(opt, 1, 3)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def load_B2AGAN_generator(opt):
    model = DTGAN(opt, 3, 1)
    if opt.cuda:
        model = model.cuda()
    else:
        model = model.cpu()

    return model


def load_A_GAN_discriminator(opt):
    if opt.dis_mode == 'vanilla_dis':
        input_shape = (opt.sar_img_channel, opt.img_height, opt.img_weight)
        model = Discriminator(input_shape)
        if opt.cuda:
            model = model.cuda()
        else:
            model = model.cpu()
    else:
        model = MsImageDiscriminator(opt.sar_img_channel)
        if opt.cuda:
            model = model.cuda()
        else:
            model = model.cpu()

    return model


def load_B_GAN_discriminator(opt):
    if opt.dis_mode == 'vanilla_dis':
        input_shape = (opt.op_img_channel, opt.img_height, opt.img_weight)
        model = Discriminator(input_shape)
        if opt.cuda:
            model = model.cuda()
        else:
            model = model.cpu()
    else:
        model = MsImageDiscriminator(opt.op_img_channel)
        if opt.cuda:
            model = model.cuda()
        else:
            model = model.cpu()

    return model


def sample_images_in_training(opt, epoch, batches_done, val_loader, G_A2B, G_B2A):
    for i, [index, batch_img1, batch_img2] in enumerate(val_loader):
        num = random.randint(0, 199)
        if i == num:
            real_A = Variable(batch_img1).cuda()
            real_B = Variable(batch_img2).cuda()
            G_A2B.eval()
            G_B2A.eval()
            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            recov_A = G_B2A(fake_B)
            recov_B = G_A2B(fake_A)

            real_A = make_grid(real_A, nrow=1, normalize=True)
            real_B = make_grid(real_B, nrow=1, normalize=True)
            fake_A = make_grid(fake_A, nrow=1, normalize=True)
            fake_B = make_grid(fake_B, nrow=1, normalize=True)
            recov_A = make_grid(recov_A, nrow=1, normalize=True)
            recov_B = make_grid(recov_B, nrow=1, normalize=True)

            image_grid = torch.cat((real_A, fake_B, recov_A, real_B, fake_A, recov_B), 1)
            save_image(image_grid, "save/vis/%s/%s/train/%s_%s_%s.png" % (opt.dataset_name, opt.model_name, epoch, batches_done, num), normalize=False)
        else:
            pass


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss