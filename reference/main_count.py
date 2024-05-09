import os
import time
import argparse
import numpy as np

import torch
from torch_geometric.data import DataLoader, Data

from load_count_data import count_data_to_tg
from models import GIN, GCN
from rnpgnn import RNPGNN

def train(train_loader, model, criterion, device, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    epoch_loss = []
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data).flatten()
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        #total_loss += float(loss) * data.num_graphs
        epoch_loss.append(loss.detach().item())
    if scheduler is not None: scheduler.step()
    #return total_loss / len(train_loader.dataset)
    return np.mean(epoch_loss)

@torch.no_grad()
def test(loader, model, criterion, device):
    model.eval()
    total_err = 0
    epoch_loss = []
    for data in loader:
        data = data.to(device)
        output = model(data).flatten()
        loss = criterion(output, data.y)
        epoch_loss.append(loss.detach().item())
        #total_err += criterion(output, data.y)
        
    #return total_err / len(loader.dataset)
    return np.mean(epoch_loss)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='triangle')
    parser.add_argument('--dataset_num', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_graphs', type=int, default=5000)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='gin')
    parser.add_argument('--rparams', type=str, default='1-1', help='recursion parameters for rnpgnn')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num_mlp_layers', type=int, default=2)
    parser.add_argument('--use_bn', action='store_true')
    parser.add_argument('--use_ln', action='store_true')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--decay_step', type=int, default=50)
    args = parser.parse_args()

    print(args)
    

    train_tg_list, val_tg_list, test_tg_list, variance = count_data_to_tg(dataset_num=args.dataset_num, task=args.task)
    train_loader = DataLoader(train_tg_list[:args.max_graphs], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_tg_list[:args.max_graphs], batch_size=args.batch_size)
    test_loader = DataLoader(test_tg_list[:args.max_graphs], batch_size=args.batch_size)

    if args.cpu: 
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    if args.model == 'gin':
        model = GIN(1, args.hidden_channels, 1, use_bn=args.use_bn, use_dropout=False, use_jk=False).to(device)
    elif args.model == 'gcn':
        model = GCN(1, args.hidden_channels, 1, use_bn=args.use_bn, use_dropout=True).to(device)
    elif args.model == 'rnpgnn':
        r = tuple(int(rk) for rk in args.rparams.split('-'))
        model = RNPGNN(r, 1, args.hidden_channels, 1, num_layers=args.num_layers, num_mlp_layers=args.num_mlp_layers, use_bn=args.use_bn, use_ln=args.use_ln, dropout=args.dropout).to(device)
    else:
        raise ValueError('Invalid model')
    print(model)
    print('Parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.decay_step == 0:
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=0.5)


    criterion = torch.nn.MSELoss()
    total_time = 0
    for epoch in range(1, args.epochs+1):
        start_time = time.time()
        loss = train(train_loader, model, criterion, device, optimizer, scheduler=scheduler)
        train_err = test(train_loader, model, criterion,  device) / variance
        val_err = test(val_loader, model, criterion,  device) / variance
        test_err = test(test_loader, model, criterion,  device) / variance
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:3d}, Loss: {loss:.4f}, Train Err: {train_err:.5f} ',
                  f'Test Err: {test_err:.5f},', f'Time: {time.time()-start_time:.3f}')
        total_time += time.time()-start_time
    print('Total time:', total_time)
    print('Time per epoch:', total_time/args.epochs)

if __name__ == '__main__':
    main()
