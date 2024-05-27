import numpy as np

from PIL import Image
from glob import glob
from os.path import join
from ntpath import basename
from metric.imqual_utils import getSSIM, getPSNR


def SSIMs_PSNRs(gtr_dir, gen_dir, im_res=(256, 256)):
    """
        - gtr_dir contain ground-truths
        - gen_dir contain generated images
    """
    gtr_paths = sorted(glob(join(gtr_dir, "*.*")))
    gen_paths = sorted(glob(join(gen_dir, "*.*")))
    ssims, psnrs = [], []
    for gtr_path, gen_path in zip(gtr_paths, gen_paths):
        gtr_f = basename(gtr_path).split('.')[0]
        gen_f = basename(gen_path).split('.')[0]

        # read images from two datasets
        r_im = Image.open(gtr_path).resize(im_res)
        g_im = Image.open(gen_path).resize(im_res)

        # get ssim on RGB channels
        ssim = getSSIM(np.array(r_im), np.array(g_im))
        ssims.append(ssim)
        # get psnt on L channel (SOTA norm)
        r_im = r_im.convert("L");
        g_im = g_im.convert("L")
        psnr = getPSNR(np.array(r_im), np.array(g_im))
        psnrs.append(psnr)
    return np.array(ssims), np.array(psnrs)