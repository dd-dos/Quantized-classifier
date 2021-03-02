import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F    
import torch.optim as optim
import os
import argparse
import numpy as np

from torchvision.models.quantization import QuantizableMobileNetV2
BEST = np.inf
def train(args):
    os.makedirs(args.cp, exist_ok=True)
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                            shuffle=True, num_workers=args.num_workers)

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                            shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    net_fp32 = QuantizableMobileNetV2(num_classes=10)
    net_fp32.train()
    net_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') #fbgemm for pc; qnnpack for mobile

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prepared_net_fp32 = torch.quantization.prepare_qat(net_fp32).to(device)
    '''
    training loop start here
    '''
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(prepared_net_fp32.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(args.num_epoches):
        running_loss = 0.0
        print("training phase:")
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = prepared_net_fp32(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 35 == 34:    
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / (35*args.batch_size)))
                running_loss = 0.0
        
        print("fp32 evaluation phase:")
        evaluation(args, prepared_net_fp32, valloader, criterion, valset, args.cp, bitwidths='fp32')
        print("int8 evaluation phase:")
        evaluation(args, torch.quantization.convert(prepared_net_fp32.to('cpu')), valloader, criterion, valset, args.cp, bitwidths='int8')
    
    '''
    training loop end here
    '''
    print('Finished Training')

    prepared_net_fp32.eval()
    net_int8 = torch.quantization.convert(prepared_net_fp32)
    torch.save(prepared_net_fp32, os.path.join(args.cp, "last_fp32.pth"))
    torch.save(net_int8, os.path.join(args.cp, "last_int8.pth"))


def evaluation(args, net, valloader, criterion, valset, checkpoint, bitwidths): 
    with torch.no_grad():
        global BEST
        num_samples = len(valset)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        net.to(device).eval()
        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # print statistics
            running_loss += loss.item()
            if i % 10 == 9:    # print every 2000 mini-batches
                print('current loss: %.3f' %
                    (running_loss / (10*args.batch_size)))
                running_loss = 0.0
        
        average_loss = running_loss/num_samples
        if average_loss < BEST:
            BEST = average_loss
            torch.save(net, os.path.join(checkpoint, "{}_best.pth".format(bitwidths)))


def argparser():
    P = argparse.ArgumentParser(description='Cifar-10 classifier')
    P.add_argument('--batch_size', type=int, required=True, help='batch size')
    P.add_argument('--num_workers', type=int, default=8, help='number of workers')
    P.add_argument('--cp', type=str, required=True, help='checkpoint')
    P.add_argument('--num_epoches', type=int, default=666, help='number of epoches')
    args = P.parse_args()

    return args

if __name__=="__main__":
    args = argparser()
    train(args)