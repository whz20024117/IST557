from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from torch.utils.data import TensorDataset, DataLoader
import torch
from torch import nn
import numpy as np


########################## Helper functions and classes ###############################
def load_data(path):
    data = np.load(path)
    x = data['x']
    y = data['y']
    locations = data['locations']
    times = data['times']

    data.close()

    return x, y, locations, times


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def export_result(fname, rmse):
    with open(fname, mode='w') as f:
        f.write("RMSE is: {:.10f}".format(rmse))


def make_torch_tensor(x, y, locations, times):
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    N = x.shape[0]
    locations = torch.tensor(locations, dtype=torch.float32).view(N, -1)
    times = torch.tensor(times, dtype=torch.float32).view(N, -1)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
        locations = locations.cuda()
        times = times.cuda()

    return x, y, locations, times


class RNNNet(nn.Module):
    def __init__(self, num_features, step, dropout=0, use_lstm=False):
        super(RNNNet, self).__init__()
        self.step = step

        if use_lstm:
            self.rnn = nn.LSTM(num_features, 256, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.RNN(num_features, 256, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        # Take last element and concat other input here
        self.fc3 = nn.Linear(16 + 3, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x, times, locations):
        assert x.shape[1] == self.step
        assert times.shape[1] + locations.shape[1] == 3

        out, _ = self.rnn(x)
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.relu(self.fc2(out))
        out = out[:, -1, :]
        out = torch.cat((out, times, locations), 1)
        out = nn.functional.relu(self.fc3(out))
        out = self.fc4(out)
        return out




############################ HW 4 starts here ######################################
def q1():
    x, y, locations, times = load_data('./data/val.npz')

    mid = np.ravel_multi_index((3, 3), (7, 7))
    # Historical average
    hist_seq = x[:, :, mid]
    y_pred = np.mean(hist_seq, axis=-1)

    print('\nUsing Historical Mean:')
    results = rmse(y, y_pred)
    print('    Root Mean Square Error is: {:.6f}'.format(results))
    export_result('./results/q1.txt', results)


def q2():
    x, y, locations, times = load_data('./data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    X_train = x.reshape(len(x), -1)
    X_val = x_val.reshape(len(x_val), -1)
    reg = LinearRegression(n_jobs=-1)
    reg.fit(X_train, y)
    pred = reg.predict(X_val)

    print('\nUsing Linear Regression:')
    print('    Root Mean Square Error is: {:.6f}'.format(rmse(y_val, pred)))
    export_result('./results/q2.txt', rmse(y_val, pred))


def q3():
    x, y, locations, times = load_data('./data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    X_train = x.reshape(len(x), -1)
    X_val = x_val.reshape(len(x_val), -1)
    reg = XGBRegressor(learning_rate=0.1)
    reg.fit(X_train, y)
    pred = reg.predict(X_val)

    print('\nUsing XGBoost:')
    print('    Root Mean Square Error is: {:.6f}'.format(rmse(y_val, pred)))
    export_result('./results/q3.txt', rmse(y_val, pred))


def q4_train():
    BATCH_SIZE = 32
    LR = 0.001
    EPOCH = 120
    save_path = './save/q4.tar'

    x, y, locations, times = load_data('./data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    x, y, locations, times = make_torch_tensor(x, y, locations, times)
    with torch.no_grad():
        x_val, y_val, loc_val, t_val = make_torch_tensor(x_val, y_val, loc_val, t_val)

    # Construct Dataloader
    dataset = TensorDataset(x, times, locations, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Construct Model
    model = RNNNet(x.shape[2], x.shape[1], use_lstm=False)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print("\nUsing RNN: ")
    best_result = 1e5
    for ep in range(EPOCH):
        print("\n    Epoch {}:".format(ep))
        iter_loader = iter(loader)
        for i in range(len(loader)):
            train_x, train_t, train_loc, train_y = next(iter_loader)

            opt.zero_grad()

            y_hat = model(train_x, train_t, train_loc)
            loss = loss_fn(y_hat, train_y)

            loss.backward()

            opt.step()

            if i % 500 == 0:
                print("    |Epoch {} Step {}|  Loss: {:.8f}, RMSE: {:.8f}".format(ep,
                                                                                  i,
                                                                                  loss.item(),
                                                                                  loss.item()**0.5))

        # Validation
        with torch.no_grad():
            pred = model(x_val, t_val, loc_val).cpu().numpy()
            result = rmse(pred, y_val.cpu().numpy())

        if save_path:
            if result < best_result:
                best_result = result
                print("    Saving... Best RMSE: {:.6f}".format(best_result))
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                }, save_path)
            else:
                print("   Not Save... Best RMSE: {:.6f}".format(best_result))

        print("    Epoch {}: RMSE is {:.8f}".format(ep, result))


def q5_train():
    BATCH_SIZE = 32
    LR = 0.001
    EPOCH = 120
    save_path = './save/q5.tar'

    x, y, locations, times = load_data('./data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    x, y, locations, times = make_torch_tensor(x, y, locations, times)
    with torch.no_grad():
        x_val, y_val, loc_val, t_val = make_torch_tensor(x_val, y_val, loc_val, t_val)

    # Construct Dataloader
    dataset = TensorDataset(x, times, locations, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Construct Model
    model = RNNNet(x.shape[2], x.shape[1], use_lstm=True)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    print("\nUsing LSTM: ")
    best_result = 1e5
    for ep in range(EPOCH):
        print("\n    Epoch {}:".format(ep))
        iter_loader = iter(loader)
        for i in range(len(loader)):
            train_x, train_t, train_loc, train_y = next(iter_loader)

            opt.zero_grad()

            y_hat = model(train_x, train_t, train_loc)
            loss = loss_fn(y_hat, train_y)

            loss.backward()

            opt.step()

            if i % 500 == 0:
                print("    |Epoch {} Step {}|  Loss: {:.8f}, RMSE: {:.8f}".format(ep,
                                                                                  i,
                                                                                  loss.item(),
                                                                                  loss.item() ** 0.5))

        # Validation
        with torch.no_grad():
            pred = model(x_val, t_val, loc_val).cpu().numpy()
            result = rmse(pred, y_val.cpu().numpy())

        if save_path:
            if result < best_result:
                best_result = result
                print("    Saving... Best RMSE: {:.6f}".format(best_result))
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                }, save_path)
            else:
                print("   Not Save... Best RMSE: {:.6f}".format(best_result))

        print("    Epoch {}: RMSE is {:.8f}".format(ep, result))


def q4q5_test(fname, save_path, use_lstm):
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    with torch.no_grad():
        x_val, y_val, loc_val, t_val = make_torch_tensor(x_val, y_val, loc_val, t_val)

    model = RNNNet(x_val.shape[2], x_val.shape[1], use_lstm=use_lstm)
    if torch.cuda.is_available():
        model.cuda()

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        model.eval()
        pred = model(x_val, t_val, loc_val).cpu().numpy()
        result = rmse(pred, y_val.cpu().numpy())

    export_result(fname, result)


class Q6Net(nn.Module):
    def __init__(self, step, edge=7, dropout=0.0):
        super(Q6Net, self).__init__()
        self.step = step
        self.edge = edge
        self.dropout = dropout
        self.conv = nn.Sequential(nn.Conv2d(1, 32, 3),
                                  nn.ReLU()
                                  )

        self.conv_fc = nn.Sequential(nn.Linear(5*5*32, 128),
                                     nn.ReLU(),
                                     nn.Dropout(p=dropout)
                                     )

        self.rnn = nn.LSTM(128, 256, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 16)
        # Take last element and concat other input here
        self.fc3 = nn.Linear(16 + 3, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x, times, locations):
        assert x.shape[1] == self.step
        assert times.shape[1] + locations.shape[1] == 3
        N = x.shape[0]

        x = x.view(-1, self.step, self.edge, self.edge)
        out = None
        for st in range(self.step):
            conv_out = self.conv(x[:, st:st+1, :, :])
            conv_out = conv_out.view(N, 1, -1)
            conv_out = self.conv_fc(conv_out)
            if out is None:
                out = conv_out
            else:
                out = torch.cat((out, conv_out), 1)

        assert out.shape[1] == self.step

        out, _ = self.rnn(out)
        out = out[:, -1, :]
        out = nn.functional.relu(self.fc1(out))
        out = nn.functional.dropout(out, self.dropout, self.training)
        out = nn.functional.relu(self.fc2(out))
        out = nn.functional.dropout(out, self.dropout, self.training)
        out = torch.cat((out, times, locations), 1)
        out = nn.functional.relu(self.fc3(out))
        out = self.fc4(out)
        return out

def q6():
    BATCH_SIZE = 32
    LR = 0.001
    EPOCH = 120
    save_path = './save/q6_3.tar'

    x, y, locations, times = load_data('./data/train.npz')
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    x, y, locations, times = make_torch_tensor(x, y, locations, times)
    with torch.no_grad():
        x_val, y_val, loc_val, t_val = make_torch_tensor(x_val, y_val, loc_val, t_val)

    # Construct Dataloader
    dataset = TensorDataset(x, times, locations, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Construct Model
    model = Q6Net(x.shape[1], edge=7, dropout=0.4)
    if torch.cuda.is_available():
        model.cuda()

    # Loss and Optimizer
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # Training loop
    best_result = 1e5
    for ep in range(EPOCH):
        model.train()
        print("\n    Epoch {}:".format(ep))
        iter_loader = iter(loader)
        for i in range(len(loader)):
            train_x, train_t, train_loc, train_y = next(iter_loader)

            opt.zero_grad()

            y_hat = model(train_x, train_t, train_loc)
            loss = loss_fn(y_hat, train_y)

            loss.backward()

            opt.step()

            if i % 500 == 0:
                print("    |Epoch {} Step {}|  Loss: {:.8f}, RMSE: {:.8f}".format(ep,
                                                                                  i,
                                                                                  loss.item(),
                                                                                  loss.item() ** 0.5))

        # Validation
        with torch.no_grad():
            model.eval()
            pred = model(x_val, t_val, loc_val).cpu().numpy()
            result = rmse(pred, y_val.cpu().numpy())

        if save_path:
            if result < best_result:
                best_result = result
                print("    Saving... Best RMSE: {:.6f}".format(best_result))
                torch.save({
                    'epoch': ep,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict()
                }, save_path)
            else:
                print("   Not Save... Best RMSE: {:.6f}".format(best_result))

        print("    Epoch {}: RMSE is {:.8f}".format(ep, result))


def q6_test(fname, save_path):
    x_val, y_val, loc_val, t_val = load_data('./data/val.npz')

    with torch.no_grad():
        x_val, y_val, loc_val, t_val = make_torch_tensor(x_val, y_val, loc_val, t_val)

    model = Q6Net(x_val.shape[1], edge=7, dropout=0.0)
    if torch.cuda.is_available():
        model.cuda()

    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        model.eval()
        pred = model(x_val, t_val, loc_val).cpu().numpy()
        result = rmse(pred, y_val.cpu().numpy())

    export_result(fname, result)




if __name__ == '__main__':
    # q1()
    # q2()
    # q3()
    # q4_train()
    # q4q5_test('./results/q4.txt', './save/q4.tar', use_lstm=False)
    # q5_train()
    # q4q5_test('./results/q5.txt', './save/q5.tar', use_lstm=True)
    q6()
    q6_test('./results/q6_3.txt', './save/q6_3.tar')

    pass
