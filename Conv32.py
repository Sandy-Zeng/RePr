import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as datasets
from torchvision import transforms
import numpy as np
from scipy.linalg import null_space
from scipy import linalg
import os
from summaries import TensorboardSummary

summary = TensorboardSummary('./repr_dir/test4/net_1')
net1_writer = summary.create_summary()
test_summary = TensorboardSummary('./repr_dir/test4/net_2')
net2_writer = test_summary.create_summary()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
        # self.avgpool = nn.AvgPool2d(32)
        self.fc = nn.Linear(32 * 32 * 32, 10)

    def forward(self, x, spasity, mode):
        # input shape: (b, 3, 32, 32)
        # if mode == 'sub-network':
        #     res = F.relu(1.0/spasity[0] * self.conv1(x))
        #     res = F.relu(1.0 / spasity[1] * self.conv2(res))
        #     res = F.relu(1.0 / spasity[2] * self.conv3(res))
        # else:
        res = F.relu(self.conv1(x))
        res = F.relu(self.conv2(res))
        res = F.relu(self.conv3(res))
        # res = self.avgpool(res).squeeze(3).squeeze(2)
        res = res.view(res.shape[0], 32 * 32 * 32)
        res = F.softmax(self.fc(res), dim=1)
        return res


def log(log_file, log):
    print(log)
    log_file.write(log + '\n')


def prepare_data():
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    kwargs = {'pin_memory': True}

    trainset = datasets.CIFAR10('../data/CIFAR10', train=True, download=True, transform=transform_train)
    testset = datasets.CIFAR10('../data/CIFAR10', train=False, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=False, **kwargs)

    return train_loader, val_loader

def train(epoch, training_type,spasity):
    # scheduler1.step()
    # scheduler2.step()
    corrected1 = 0
    corrected2 = 0
    total = 0
    log(log_file, "Epoch: {}, ".format(epoch) + training_type)
    for i, (input, target) in enumerate(train_loader):
        # if training_type == 'sub-network':
        #     drop()

        input = input.to(device)
        target = target.to(device)
        res1 = net1(input,spasity=spasity,mode='full-network')
        res2 = net2(input, spasity=spasity, mode=training_type)

        loss1 = criterion(res1, target)
        loss2 = criterion(res2, target)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        if training_type == 'sub-network':
            zero_grad()

        total += input.size(0)

        _, predicted1 = torch.max(res1.data, 1)
        corrected1 += (predicted1 == target).sum()

        _, predicted2 = torch.max(res2.data, 1)
        corrected2 += (predicted2 == target).sum()
        if i % 1000 == 0:
            log(log_file,
                "[{}/{}], Standard Loss: {}, RePr Loss: {}".format(i, len(train_loader), loss1.item(), loss2.item()))
    acc1 = corrected1.item() / total
    acc2 = corrected2.item() / total
    log(log_file, "Training, Standard Accuracy: {}, RePr Accuracy: {}".format(acc1, acc2))
    net1_writer.add_scalar('train acc',acc1,epoch)
    net2_writer.add_scalar('train acc',acc2,epoch)

net1_val_acc = 0.
net2_val_acc = 0.
def test(training_type,spasity):
    global net1_val_acc,net2_val_acc
    corrected1 = 0
    corrected2 = 0
    total = 0
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        target = target.to(device)
        res1 = net1(input, spasity=spasity, mode='full-network')
        res2 = net2(input, spasity=spasity, mode=training_type)
        # res1 = net1(input)
        # res2 = net2(input)

        total += input.size(0)

        _, predicted1 = torch.max(res1.data, 1)
        corrected1 += (predicted1 == target).sum()

        _, predicted2 = torch.max(res2.data, 1)
        corrected2 += (predicted2 == target).sum()
    acc1 = corrected1.item() / total
    acc2 = corrected2.item() / total
    log(log_file, "Testing , Standard Accuracy: {}, RePr Accuracy: {}".format(acc1, acc2))
    log(log_file, "\n")
    net1_writer.add_scalar('val acc', acc1, epoch)
    net2_writer.add_scalar('val acc', acc2, epoch)
    if acc1 >net1_val_acc:
        net1_val_acc = acc1
    if acc2 > net2_val_acc:
        net2_val_acc = acc2


def ortho(conv_layer):
    # c_out, c_in, k_w, k_h = conv_layer.shape
    # conv_layer = conv_layer.view(c_out, -1)
    c_out = conv_layer.shape[0]
    normalized_conv_layer = conv_layer / torch.sqrt(torch.sum(torch.pow(conv_layer, 2), dim=1)).unsqueeze(1).expand(
        conv_layer.shape)
    P = torch.abs(
        torch.matmul(normalized_conv_layer, normalized_conv_layer.transpose(0, 1)) - torch.eye(c_out).to(device))
    assert (P.shape == (c_out, c_out))
    rank = torch.sum(P, dim=1) / c_out
    return rank.view(c_out, 1)


def cal_dropped_filters(p):
    ranks = []
    for layer_index, v in conv_layers.items():
        rank = ortho(v.view(v.size(0), -1))
        for filter_index in range(rank.shape[0]):
            ranks.append((layer_index, filter_index, rank[filter_index].data.cpu().numpy()[0]))
    ranks.sort(key=lambda x: x[2])
    ranks = ranks[-int(len(ranks) * p):]
    dropped_filters = dict()
    dropped_filters_weight = dict()
    param = list(net2.parameters())
    for r in ranks:
        layer_index = r[0]
        filter_index = r[1]
        if layer_index not in dropped_filters:
            dropped_filters[layer_index] = []
            dropped_filters_weight[layer_index] = []
            dropped_filters[layer_index].append(filter_index)
            dropped_filters_weight[layer_index].append(param[layer_index][filter_index])
        else:
            dropped_filters[layer_index].append(filter_index)
            dropped_filters_weight[layer_index].append(param[layer_index][filter_index])

    for layer, weight_list in dropped_filters_weight.items():
        dropped_filters_weight[layer] = torch.cat([w.flatten().unsqueeze(0) for w in weight_list], dim=0)

    spasity = [1.0,1.0,1.0]
    for layer_index, filter_list in dropped_filters.items():
        print ('layer %d drop %d filter' % (layer_index, len(filter_list)))
        if layer_index == 0:
            spasity[0] = len(filter_list)/32.0
        if layer_index == 2:
            spasity[1] = len(filter_list)/32.0
        if layer_index == 4:
            spasity[2] = len(filter_list)/32.0
    return dropped_filters, dropped_filters_weight,spasity


def drop():
    param = list(net2.parameters())
    # print (net2.parameters())
    # print (param)
    # assert False
    for layer, filter_list in dropped_filters.items():
        for f in filter_list:
            # param[layer][f] = torch.zeros(param[layer][f].shape).to(device)
            torch.nn.init.constant_(param[layer][f], 0.0)

def zero_grad():
    param = list(net2.parameters())
    for layer, filter_list in dropped_filters.items():
        for f in filter_list:
            param[layer].grad[f]= 0.
            # print ('zero grad')
            # param[layer][f] = torch.zeros(param[layer][f].shape).to(device)
            # torch.nn.init.constant_(param[layer][f], 0.0)



def re_initiate_zyy():
    param = list(net2.parameters())
    for layer_index, dropped_filter_list in dropped_filters.items():
        c_out, c_in, k_w, k_h = param[layer_index].shape
        conv_layer = param[layer_index].view(c_out, -1).detach().cpu().numpy()

        undropped_filter_list = list(set(np.arange(c_out)) - set(dropped_filter_list))
        # undropped_filters = conv_layer[undropped_filter_list]
        undropped_filters = torch.FloatTensor(conv_layer[undropped_filter_list]).cuda()

        dropped_w = dropped_filters_weight[layer_index].detach()
        W = torch.cat([undropped_filters, dropped_w], dim=0)
        W = W.cpu().numpy()

        if len(undropped_filter_list) != 0:
            q, r = linalg.qr(W, mode='full')
            M, N = r.shape
            # print(M,N)
            K = len(dropped_filter_list)
            # new_ortho_filter = torch.FloatTensor((q.dot(torch.eye(N, K))).T)
            if M >= N:
                if len(undropped_filter_list) < N:
                    # print(undropped_filters.shape)
                    q1,r1 = linalg.qr(undropped_filters.cpu().numpy(),mode='full')
                    # print(r1.shape)
                    # print(K)
                    new_ortho_filter = torch.from_numpy((null_space(r1)).T)
                    print(torch.matmul(undropped_filters,torch.transpose(new_ortho_filter.cuda(),0,1)))
                else:
                    ranks = ortho(undropped_filters)
                    ranks = [(i, ranks[i]) for i in range(ranks.shape[0])]
                    ranks.sort(key=lambda x: x[1])
                    ranks = ranks[-N+1:]
                    select_filter = [r[0] for r in ranks]
                    q1, r1 = linalg.qr(undropped_filters[select_filter].cpu().numpy(), mode='full')
                    new_ortho_filter = torch.from_numpy((null_space(r1)).T)
                    print(torch.matmul(undropped_filters, torch.transpose(new_ortho_filter.cuda(), 0, 1)))
            else:
                new_ortho_filter = torch.from_numpy((null_space(r)).T)
                print(torch.matmul(undropped_filters, torch.transpose(new_ortho_filter.cuda(), 0, 1)))

            if new_ortho_filter.shape[0] < K:
                K1,K2 = new_ortho_filter.shape
                zero_pad = torch.zeros((K-K1,K2))
                new_ortho_filter = torch.cat([new_ortho_filter,zero_pad],dim=0)
            if new_ortho_filter.shape[0] > K:
                new_ortho_filter = new_ortho_filter[:K]
            # print(new_ortho_filter.shape)
            maxValue = torch.max(undropped_filters)
            minValue = torch.min(undropped_filters)
            new_ortho_filter = torch.clamp(new_ortho_filter,min=minValue,max=maxValue).cuda()
            new_ortho_filter = (torch.mean(undropped_filters)/torch.mean(new_ortho_filter)) * new_ortho_filter
            print (torch.mean(undropped_filters))
            print (torch.mean(new_ortho_filter))
            # print ('-----------------')
            # print (new_ortho_filter)
            torch.nn.init.constant(param[layer_index][dropped_filter_list], 0)
            param[layer_index][dropped_filter_list].data += new_ortho_filter.view(K, c_in, k_w, k_h).float().cuda()
        else:
            torch.nn.init.orthogonal_(param[layer_index])


def re_initiate_hzb():
    param = list(net2.parameters())
    for layer_index, dropped_filter_list in dropped_filters.items():
        c_out, c_in, k_w, k_h = param[layer_index].shape
        conv_layer = param[layer_index].view(c_out, -1).detach().cpu().numpy()

        undropped_filter_list = list(set(np.arange(c_out)) - set(dropped_filter_list))
        undropped_w = torch.FloatTensor(conv_layer[undropped_filter_list]).cuda()
        dropped_w = dropped_filters_weight[layer_index].detach()
        W = torch.cat([undropped_w, dropped_w], dim=0)

        if len(undropped_filter_list) != 0:
            filter_num, filter_size = W.shape

            # null space not exist
            if filter_num >= filter_size:
                if len(undropped_filter_list) < filter_size:
                    ns = null_space(undropped_w)
                else:  # select 5 filters that are most overlapping
                    ranks = ortho(undropped_w)
                    ranks = [(i, ranks[i]) for i in range(ranks.shape[0])]
                    ranks.sort(key=lambda x: x[1])
                    ranks = ranks[-5:]
                    select_filter = [r[0] for r in ranks]
                    ns = null_space(undropped_w[select_filter])
            else:
                ns = null_space(W)

            ortho_basis_num = ns.shape[1]
            for drop_f in dropped_filter_list:
                rand = np.random.rand(ortho_basis_num)
                new_ortho_filter = torch.from_numpy(ns.dot(rand)).to(device)
                torch.nn.init.constant_(param[layer_index][drop_f], 0)
                param[layer_index][drop_f].data += new_ortho_filter.view(c_in, k_w, k_h).float().cuda()
        else:
            torch.nn.init.orthogonal_(param[layer_index])


device = torch.device("cuda")
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
train_loader, val_loader = prepare_data()

net1 = Net().to(device)
net2 = Net().to(device)

criterion = nn.CrossEntropyLoss()

optimizer1 = optim.SGD(params=net1.parameters(), lr=0.01)
optimizer2 = optim.SGD(params=net2.parameters(), lr=0.01)
# scheduler1 = optim.lr_scheduler.MultiStepLR(optimizer1, milestones=[40, 70], gamma=0.1)
# scheduler2 = optim.lr_scheduler.MultiStepLR(optimizer2, milestones=[40, 70], gamma=0.1)

conv_layers = dict()
cnt = 0
for name, param in net2.named_parameters():
    if 'weight' in name and param.requires_grad == True and 'fc' not in name:
        conv_layers[cnt] = param
    cnt += 1

N = 3
S1 = 20
S2 = 10
p = 0.3

open_type = 'a' if os.path.exists('log.txt') else 'w'
log_file = open('log.txt', open_type)
epoch = 0
while epoch < 109:
    # train(epoch, 'full network')
    # test()
    # epoch += 1
    for e in range(S1):
        if epoch > 109:
            break
        spasity = [1.0,1.0,1.0]
        train(epoch, 'full network',spasity)
        test('full network',spasity)
        epoch += 1

    dropped_filters, dropped_filters_weight,spasity = cal_dropped_filters(p)
    drop()

    for e in range(S2):
        # drop()
        if epoch > 109:
            break
        train(epoch, 'sub-network',spasity)
        test('sub-network',spasity)
        epoch += 1

    re_initiate_zyy()
    # re_initiate_hzb()
print ('Std Train:',net1_val_acc)
print ('RePr:',net2_val_acc)
