import argparse
import torch
import utils
from torch.utils.data import DataLoader
import numpy as np
import itertools
from torchvision.utils import save_image
import os
from torch.autograd import Variable


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.0001, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=5, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs/test-field', help='base directory to save logs')
parser.add_argument('--data_root', default='./videos', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--content_dim', type=int, default=128, help='size of the content vector')
parser.add_argument('--pose_dim', type=int, default=10, help='size of the pose vector')
parser.add_argument('--image_width', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--sd_weight', type=float, default=0.0001, help='weight on adversarial loss')
parser.add_argument('--sd_nf', type=int, default=100, help='number of layers')
parser.add_argument('--content_model', default='dcgan_unet', help='model type (dcgan | dcgan_unet | vgg_unet)')
parser.add_argument('--pose_model', default='dcgan', help='model type (dcgan | unet | resnet)')
parser.add_argument('--data_threads', type=int, default=24, help='number of parallel data loading threads')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--data_type', default='drnet', help='speed up data loading for drnet training')
opt = parser.parse_args()

train_data, test_data = utils.load_dataset(opt)

test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)


def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
testing_batch_generator = get_testing_batch()


def plot_rec(x, netEC, netEP, netD):
    x_c = x[0]
    x_p = x[np.random.randint(1, opt.max_step)]

    h_c = netEC(x_c)
    h_p = netEP(x_p)

    # print('h_c shape: ', h_c.shape)
    # print('h p shape: ', h_p.shape)
    rec = netD([h_c, h_p])

    x_c, x_p, rec = x_c.data, x_p.data, rec.data
    fname = '%s/rec/rec_test.png' % (opt.log_dir)

    comparison = None
    for i in range(len(x_c)):
        if comparison is None:
            comparison = torch.stack([x_c[i], x_p[i], rec[i]])
        else:
            new_comparison = torch.stack([x_c[i], x_p[i], rec[i]])
            comparison = torch.cat([comparison, new_comparison])
    print('comparison: ', comparison.shape)

    # row_sz = 5
    # nplot = 20
    # for i in range(0, nplot - row_sz, row_sz):
    #     row = [[xc, xp, xr] for xc, xp, xr in zip(x_c[i:i + row_sz], x_p[i:i + row_sz], rec[i:i + row_sz])]
    #     print('row: ', row)
    #     to_plot.append(list(itertools.chain(*row)))
    # print(len(to_plot[0]))
    # utils.save_tensors_image(fname, comparison)
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    save_image(comparison.cpu(), fname, nrow=3)


def plot_analogy(x, netEC, netEP, netD):
    x_c = x[0]

    h_c = netEC(x_c)
    nrow = opt.batch_size
    row_sz = opt.max_step
    to_plot = []
    zeros = torch.zeros(opt.channels, opt.image_width, opt.image_width)
    # to_plot.append(zeros)
    for i in range(nrow):
        to_plot.append(x[0][i])
    to_plot = torch.stack(to_plot)

    for j in range(0, row_sz):
        h_p = netEP(x[j])
        for i in range(nrow):
            h_p[i] = h_p[0]
        rec = netD([h_c, Variable(h_p)])
        print('rec shape: ', rec.shape)
        to_plot = torch.cat([to_plot, rec])

    originals = []
    for i in range(row_sz):
        originals.append(x[i][0])
    originals = [x[0][0]] + originals

    print('original: ', len(originals))
    print('len(to_plot): ', len(to_plot))

    plt_list = []
    for i in range(len(to_plot)):
        if i % nrow == 0:
            print('int(i / nrow): ', int(i / nrow))
            plt_list.append(originals[int(i / nrow)])
        plt_list.append(to_plot[i])

    to_plot = torch.stack(plt_list)

    fname = '%s/rec/analogy_test.png' % (opt.log_dir)
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    save_image(to_plot, fname, nrow=(nrow+1))


model_path = './logs/kth128x128/2019-01-14-18-14-45/content_model=dcgan_unet-pose_model=dcgan-content_dim=128-pose_dim=10-max_step=20-sd_weight=0.000-lr=0.002-sd_nf=100-normalize=False'
has_cuda = torch.cuda.is_available()

if has_cuda:
    dtype = torch.cuda.FloatTensor
else:
    dtype = torch.FloatTensor

if has_cuda:
    checkpoint = torch.load('%s/model.pth' % model_path)
else:
    checkpoint = torch.load('%s/model.pth' % model_path, map_location='cpu')

# print(checkpoint)
netD = checkpoint['netD']
netEP = checkpoint['netEP']
netEC = checkpoint['netEC']

x = next(testing_batch_generator)
# plot_rec(x, netEC, netEP, netD)
plot_analogy(x, netEC, netEP, netD)
# print('x_c: ', x_c.shape)

