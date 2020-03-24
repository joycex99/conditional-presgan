import argparse
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable

from model import TCN

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


PATH = 'histdata/aggr_60min/'


def concat_data(path):
    """Return concatenated data from arrays.

    Args:
        path : str
            path to load data from
    Returns:
        arrays concatenated by year and pair into large aggregate array
    """
    pairs = ['EURGBP', 'EURUSD', 'USDHUF', 'USDJPY']
    final_arrs = []
    for year in range(2013, 2020):
        pair_arr = np.concatenate([np.load(
            path + pair + '_' + str(year) + '_aggregated.npy') for pair in pairs], axis=1)
        final_arrs.append(pair_arr)
    return np.concatenate(final_arrs, axis=0)


def moving_average(x):
    """Returns moving average of a window sized 3 on array x."""
    return np.convolve(x, np.ones(3), 'valid') / 3  # 3 is window size


def reshape_data_TCN(input_data, num_timesteps=30, skip_interval=30):
    """Reshapes input data into features and labels.

    Args:
        input_data : numpy array
            Input data to be reshaped
        num_timesteps : int
            Number of time steps to slice on
        skip_interval : int
            Interval at the end of the data to skip
    Returns:
        Reshaped feature and label numpy arrays
    """
    # TCN Input must be num_samples, num_timesteps, num_features
    reshaped_X = []
    reshaped_Y = []

    # Create one datapoint based on num_timesteps
    for x in range(0, input_data.shape[0] - num_timesteps, skip_interval):
        data_slice = np.array(input_data[x:x+num_timesteps, :])
        #data_slice = np.apply_along_axis(moving_average, 0, data_slice)
        reshaped_X.append(data_slice)

        # If EUR/USD price goes up in the next day pos class, else neg class
        if input_data[x+num_timesteps, 3] > input_data[x+num_timesteps - 1, 3]:
            reshaped_Y.append(1)
        else:
            reshaped_Y.append(0)

    return np.array(reshaped_X), np.array(reshaped_Y)


def train_epoch(model, X, Y, opt, device, batch_size=10, clip=1.0):
    """Trains the model for one epoch.

    Args:
        model : torch.model
            Model to train on
        X : np.array
            Feature array to train the model on.
        Y : np.array
            Label array.
        clip : np.float64
            default 1.0, Float argument for clipping
        opt : torch.optim.optimizer
            Torch optimizer to use.
        device : torch.device
            Torch device to use (cpu or gpu)
        batch_size : int
            Batch size to train on.
    Returns:
        losses : list
            List of losses accrued over the epoch.
        avg_loss : float
            Average model loss of the epoch
        accuracy : float
            Accuracy of the model
    """
    nll_loss = F.nll_loss

    if torch.cuda.is_available():
        model.cuda()

    model.train()
    losses = []
    correct = 0
    for beg_i in tqdm(range(0, X.shape[0], batch_size)):
        # Pull batch
        x_batch = X[beg_i:beg_i + batch_size, :, :]
        y_batch = Y[beg_i:beg_i + batch_size]

        # Format np arrays
        x_batch = torch.from_numpy(x_batch).float().to(device=device)
        y_batch = torch.from_numpy(y_batch).long().to(device=device)
        x_batch, y_batch = Variable(x_batch), Variable(y_batch)

        # compute loss + grad + update
        opt.zero_grad()
        y_hat = model(x_batch)
        loss = nll_loss(y_hat, y_batch)
        loss.backward()

        clip_grad_norm_(model.parameters(), clip)
        opt.step()

        # Train accuracy
        pred = y_hat.data.max(1, keepdim=True)[1]

        if torch.cuda.is_available():
            correct += pred.eq(y_batch.data.view_as(pred)).cuda().sum()
        else:
            correct += pred.eq(y_batch.data.view_as(pred)).cpu().sum()

        # save losses
        if torch.cuda.is_available():
            losses.append(loss.data.cpu().numpy())
        else:
            losses.append(loss.data.numpy())

    print('\Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        np.mean(losses), correct, X.shape[0],
        100. * correct / X.shape[0]))
    return losses, np.mean(losses), (correct / X.shape[0])


def test(model, X, Y, device, batch_size=10):
    """Evaluates the model.

    Args:
        model : torch.model
            Model to test on
        X : np.array
            Feature array to test the model on.
        Y : np.array
            Label array.
        batch_size : int
            Batch size to test on.
    Returns:
        avg_test_loss : float
            Average model loss on test dataset
        accuracy : float
            Accuracy of the model
    """
    nll_loss = F.nll_loss

    if torch.cuda.is_available():
        model.cuda()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for beg_i in range(0, X.shape[0], batch_size):
            # Pull batch
            x_batch = X[beg_i:beg_i + batch_size, :, :]
            y_batch = Y[beg_i:beg_i + batch_size]

            # Format np arrays
            x_batch = torch.from_numpy(x_batch).float().to(device=device)
            y_batch = torch.from_numpy(y_batch).long().to(device=device)
            x_batch, y_batch = Variable(
                x_batch, volatile=True), Variable(y_batch)

            # Compute accuracy
            y_hat = model(x_batch)
            test_loss += nll_loss(y_hat, y_batch, size_average=False).item()
            pred = y_hat.data.max(1, keepdim=True)[1]

            if torch.cuda.is_available():
                correct += pred.eq(y_batch.data.view_as(pred)).cuda().sum()
            else:
                correct += pred.eq(y_batch.data.view_as(pred)).cpu().sum()

        avg_test_loss = test_loss/X.shape[0]
        print('\Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avg_test_loss, correct, X.shape[0],
            100. * correct / X.shape[0]))
        return avg_test_loss, (correct / X.shape[0])


def create_directory(model_dir):
    """Creates a new directory to save model info.

    Args:
        model_dir : str
            Global directory to store all model weights and loss info
    Returns:
        dir_name : str
            Timestamped directory to save current run info
    """
    today = datetime.now()
    curr_time = today.strftime('%d%m%Y_%H%M')
    dir_name = 'model_dir/model_state_' + curr_time
    os.mkdir(dir_name)
    return dir_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', dest='modeldir',
                        type=str, help='Directory Path to save model to.', default='model_dir')
    parser.add_argument('--epochs', dest='num_epochs',
                        type=int, help='Number of epochs.', default=10)
    parser.add_argument('--dropout', dest='dropout', default=0,
                        type=float, help='dropoutval')
    parser.add_argument('--split', dest='train_split', default=0.8,
                        type=float, help='training split')
    parser.add_argument('--clip', dest='clip', default=1.0,
                        type=float, help='Gradient clip value')
    parser.add_argument('--lr', dest='lr', default=5e-04,
                        type=float, help='Learning rate')
    parser.add_argument('--datapath', dest='datapath', type=str,
                        default='oanda_data.npy', help='Path to load data')

    args = parser.parse_args()

    torch.manual_seed(1111)

    # set device for cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device' + 'cuda is avail' + str(torch.cuda.is_available()))

    # Load Raw Data
    data_arr = concat_data(PATH)

    # Reshape
    n_train = round(args.train_split * data_arr.shape[0])
    n_val = round((0.5*(1-args.train_split)) * data_arr.shape[0])
    X_train, y_train = reshape_data_TCN(data_arr[:n_train])
    X_val, y_val = reshape_data_TCN(data_arr[n_train:])
    #X_test, y_test = reshape_data_TCN(data_arr[n_train + n_val:])

    # Class distribution
    print(np.unique(y_train, return_counts=True))
    print(np.unique(y_val, return_counts=True))
    #print(np.unique(y_test, return_counts=True))

    # ------------------

    channel_sizes = [25] * 12

    # Init model
    model = TCN(30, 2, channel_sizes,
                kernel_size=6,
                dropout=args.dropout).to(device)

    # Opt/Loss
    # TODO : make this into argument instead of hardcoding
    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)

    # opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    print('Model State')
    for param_tensor in model.state_dict():
        print(param_tensor, '\t', model.state_dict()[param_tensor].size())

    print("Optimizer's state_dict:")
    for var_name in opt.state_dict():
        print(var_name, '\t', opt.state_dict()[var_name])

    e_losses = []
    avg_train_losses = []
    avg_test_losses = []
    avg_train_accs = []
    avg_test_accs = []

    try:
        print('Training for ' + str(args.num_epochs) + 'epochs')
        for e in range(args.num_epochs):
            tloss, avg_train_loss, avg_train_acc = train_epoch(
                model, X_train, y_train, opt, device)
            avg_test_loss, avg_test_acc = test(model, X_val, y_val, device)

            e_losses += tloss
            avg_train_losses.append(avg_test_loss)
            avg_test_losses.append(avg_test_loss)
            avg_train_accs.append(avg_train_acc)
            avg_test_accs.append(avg_test_acc)
    except KeyboardInterrupt:
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
        pass

    curr_dir_name = create_directory(args.modeldir)

    print('Saving model to:' + curr_dir_name+'/model')
    torch.save(model.state_dict(), curr_dir_name+'/model')

    plt.plot(avg_train_accs, label='train')
    plt.plot(avg_test_accs, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Average accuracy')
    plt.legend()
    plt.savefig(curr_dir_name + '/acc.png')
    plt.close()

    plt.plot(avg_train_losses, label='train')
    plt.plot(avg_test_losses, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.savefig(curr_dir_name + '/loss.png')
    plt.show()
