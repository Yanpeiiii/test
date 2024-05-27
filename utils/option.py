import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    # base options
    parser.add_argument("--cuda", type=bool, default=True,
                        help="have GPU or not")
    parser.add_argument("--seed", type=int, default=2023,
                        help="seed for random")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=400,
                        help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default=" ",
                        help="name of the dataset")
    parser.add_argument("--dataset_dir", type=str, default=" ",
                        help="dataset dir")
    parser.add_argument("--model_name", type=str, default=" ",
                        help="name of the model")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="size of the batches")
    parser.add_argument("--lr_gen", type=float, default=0.00005,
                        help="adam: learning rate")
    parser.add_argument("--lr_dis", type=float, default=0.0001,
                        help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5,
                        help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999,
                        help="adam: decay of second order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=200,
                        help="epoch from which to start lr decay")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256,
                        help="size of image height")
    parser.add_argument("--img_weight", type=int, default=256,
                        help="size of image width")
    parser.add_argument("--op_img_channel", type=int, default=3,
                        help="number of optical image channels")
    parser.add_argument("--sar_img_channel", type=int, default=1,
                        help="number of sar image channels")
    parser.add_argument("--sample_interval", type=int, default=400,
                        help="interval between saving generator outputs")
    parser.add_argument("--checkpoint_interval", type=int, default=-1,
                        help="interval between saving model checkpoints")
    parser.add_argument("--n_residual_blocks", type=int, default=9,
                        help="number of residual blocks in generator")
    parser.add_argument("--lambda_gen", type=float, default=1.0,
                        help="gen loss weight")
    parser.add_argument("--lambda_cyc", type=float, default=10.0,
                        help="cycle loss weight")
    parser.add_argument("--lambda_id", type=float, default=3.0,
                        help="identity loss weight")
    parser.add_argument("--fid_batchsize", type=int, default=20,
                        help="size of the calculate fid score")
    parser.add_argument("--fid_dims", type=int, default=2048,
                        help="dim of the calculate fid score")
    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
    parser.add_argument("--dis_mode", type=str, default="msimage_dis",
                        help="vanilla_dis or msimage_dis")

    parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                        help='scaling and cropping of images at load time [resize_and_crop | none]')
    parser.add_argument('--angle', type=int, default=30,
                        help='rotate angle')
    parser.add_argument('--load_size', type=int, default=300,
                        help='scale images to this size')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='then crop to this size')
    parser.add_argument('--no_flip', type=bool, default=True,
                        help='if specified, do not flip(left-right) the images for data augmentation')
    parser.add_argument('--phase', type=str, default='train',
                        help='train, val, test, etc')

    # For channel transformer
    parser.add_argument('--channel_embeddings_dropout_rate', type=float, default=0.1,
                        help='')
    parser.add_argument('--channel_transformer_KV_size', type=int, default=960,
                        help='')
    parser.add_argument('--channel_transformer_num_heads', type=int, default=4,
                        help='')
    parser.add_argument('--channel_attention_dropout_rate', type=float, default=0.1,
                        help='')
    parser.add_argument('--channel_mlp_dropout_rate', type=float, default=0.0,
                        help='')
    parser.add_argument('--channel_transformer_expand_ratio', type=int, default=4,
                        help='')
    parser.add_argument('--channel_transformer_num_layers', type=int, default=4,
                        help='')

    opt = parser.parse_args()

    return opt