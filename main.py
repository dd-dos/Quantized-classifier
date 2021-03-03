import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F    
import torch.optim as optim
import os
import argparse
import numpy as np

from torchvision.models.quantization import mobilenet_v2, QuantizableMobileNetV2
from tqdm import tqdm
BEST_ACC = 0

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
    
    net_fp32 = mobilenet_v2(num_classes=10)
    net_fp32.train()
    net_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') #fbgemm for pc; qnnpack for mobile
    torch.backends.quantized.engine='fbgemm'

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prepared_net_fp32 = torch.quantization.prepare_qat(net_fp32).to(device)
    prepared_net_fp32.load_state_dict(torch.load("/content/drive/MyDrive/training/Quantized-classifier/fp32_best.pth"))
    '''
    training loop start here
    '''
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(prepared_net_fp32.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(args.num_epoches):
        running_loss = 0.0
        counter = 0.0
        print("=> training phase:")
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
            outputs = outputs.detach().cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            labels = labels.cpu().numpy()
            diff = outputs - labels
            counter += len(np.where(diff==0)[0])

            if i % 35 == 34:    
                accuracy = counter / (35*args.batch_size)
                print('[%d, %5d] loss: %.3f - acc: %.3f' %
                    (epoch + 1, i + 1, running_loss / (35*args.batch_size), accuracy))
                running_loss = 0.0
                counter = 0

        print("=> int8 evaluation phase:")
        net_int8 = torch.quantization.convert(prepared_net_fp32.cpu().eval())
        evaluation(args, net_int8, valloader, criterion, valset, args.cp, bitwidths='int8')
        print("=> fp32 evaluation phase:")
        evaluation(args, prepared_net_fp32, valloader, criterion, valset, args.cp, bitwidths='fp32')

    
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
        global BEST_ACC
        num_samples = len(valset)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        running_loss = 0.0
        counter = 0.0

        for i, data in enumerate(valloader, 0):
            inputs, labels = data

            if bitwidths=='fp32':
                inputs = inputs.to(device)
                labels = labels.to(device)
                net = net.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            outputs = outputs.cpu().numpy()
            outputs = np.argmax(outputs, axis=1)
            labels = labels.cpu().numpy()
            diff = outputs - labels
            counter += len(np.where(diff==0)[0])

        accuracy = counter/num_samples
        average_loss = running_loss/num_samples
        print("==> val loss: {:.3f} - val acc: {:.3f}".format(average_loss, accuracy))
        if accuracy > BEST_ACC:
            BEST_ACC = accuracy
            print("===> saving model at {}".format(checkpoint))
            torch.save(net.state_dict(), os.path.join(checkpoint, "{}_best.pth".format(bitwidths)))


def test_qtmodel(checkpoint, split='val'):
    net_fp32 = mobilenet_v2(num_classes=10)
    net_fp32.train()
    net_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') #fbgemm for pc; qnnpack for mobile
    torch.backends.quantized.engine='fbgemm'
    prepared_net_fp32 = torch.quantization.prepare_qat(net_fp32)
    prepared_net_fp32.load_state_dict(torch.load("/content/drive/MyDrive/training/Quantized-classifier/fp32_best.pth"))
    net_int8 = torch.quantization.convert(prepared_net_fp32.cpu().eval())
    # net_int8.load_state_dict(torch.load(checkpoint))
    # print(torch.load(checkpoint))
    net_int8.eval()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                            shuffle=False, num_workers=8)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=8)
    if split == 'train':
        loader = trainloader
        dataset = trainset
    else:
        loader = valloader
        dataset = valset

    with torch.no_grad():
        num_samples = len(dataset)
        counter = 0
        for i, data in tqdm(enumerate(loader, 0)):
            inputs, labels = data
            out = net_int8(inputs).cpu().numpy()
            out = np.argmax(out, axis=1)

            labels = labels.cpu().numpy()
            diff = out - labels
            counter += len(np.where(diff==0)[0])
    return counter/num_samples*100


def test_fp32_model(checkpoint, split='val'):
    net_fp32 = mobilenet_v2(num_classes=10)
    net_fp32.train()
    net_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm') #fbgemm for pc; qnnpack for mobile
    torch.backends.quantized.engine='fbgemm'
    net_fp32 = torch.quantization.prepare_qat(net_fp32)
    net_fp32.load_state_dict(torch.load(checkpoint))
    net_fp32.eval()

    transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    valset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64,
                                            shuffle=False, num_workers=8)
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=8)

    if split == 'train':
        loader = trainloader
        dataset = trainset
    else:
        loader = valloader
        dataset = valset

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net_fp32.to(device)

    with torch.no_grad():
        num_samples = len(dataset)
        counter = 0
        for i, data in tqdm(enumerate(loader, 0)):
            inputs, labels = data
            inputs = inputs.to(device)
            out = net_fp32(inputs).cpu().numpy()
            out = np.argmax(out, axis=1)

            labels = labels.cpu().numpy()
            diff = out - labels
            counter += len(np.where(diff==0)[0])
    return counter/num_samples*100

def argparser():
    P = argparse.ArgumentParser(description='Cifar-10 classifier')
    P.add_argument('--batch_size', type=int, required=True, help='batch size')
    P.add_argument('--num_workers', type=int, default=8, help='number of workers')
    P.add_argument('--cp', type=str, required=True, help='checkpoint')
    P.add_argument('--num_epoches', type=int, default=666, help='number of epoches')
    args = P.parse_args()

    return args

if __name__=="__main__":
    # args = argparser()
    # train(args)
    print(test_qtmodel("/content/drive/MyDrive/training/Quantized-classifier/int8_best.pth", split='train'))
    # print(test_fp32_model("/content/drive/MyDrive/training/Quantized-classifier/fp32_best.pth"))