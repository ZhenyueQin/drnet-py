import models.lstm as models
import argparse
import torch
import utils
from torch.utils.data import DataLoader
from collections import OrderedDict

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--model_path', default='.', help='path to drnet model')
parser.add_argument('--data_root', default='./videos', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--n_past', type=int, default=10, help='number of frames to condition on')
parser.add_argument('--n_future', type=int, default=10, help='number of frames to predict')
parser.add_argument('--rnn_size', type=int, default=256, help='dimensionality of hidden layer')
parser.add_argument('--rnn_layers', type=int, default=2, help='number of layers')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--data_threads', type=int, default=5, help='number of parallel data loading threads')
parser.add_argument('--data_type', default='sequence', help='speed up data loading for drnet training')

opt = parser.parse_args()
opt.max_step = opt.n_past+opt.n_future
opt.pose_dim = 5
opt.content_dim = 128

train_data, test_data = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=opt.data_threads,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)
test_loader = DataLoader(test_data,
                         num_workers=opt.data_threads,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

lstm = models.lstm(opt.pose_dim+opt.content_dim, opt.pose_dim, opt.rnn_size, opt.rnn_layers, opt.batch_size, opt.normalize)
lstm_dict = torch.load('pretrained_models/kth128x128_model.pth', map_location='cpu')
new_state_dict = OrderedDict()
# print(lstm_dict)
for k, v in lstm_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v
print(new_state_dict)
lstm.load_state_dict(new_state_dict)


def get_testing_batch(dtype=torch.cuda.FloatTensor):
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
testing_batch_generator = get_testing_batch()


lstm.eval()
for epoch in range(opt.niter):
    x = next(testing_batch_generator)
