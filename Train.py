from MyNet import NeuralNet
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from MyData import MyDatasets

class Train:

    def __init__(self):
        self.net = NeuralNet().cuda()
        self.costfunc = nn.MSELoss()
        self.opt = torch.optim.Adam(self.net.parameters())
        self.datasets = MyDatasets("img")

    def loaderdata(self):
        return  DataLoader(dataset=self.datasets,batch_size=600,shuffle=True)

    def train(self):
        trainlist = self.loaderdata()
        for j in range(20):
            print("epochs:{}".format(j))
            for i,(input, target) in enumerate(trainlist):
                out = self.net(input.cuda())
                target = target.cuda()
                target = torch.zeros(target.size(0),2).cuda().scatter(1, target, 1)
                loss = self.costfunc(out,target)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if i % 5 ==0:
                    print("{}/{},loss:{}".format(i,len(trainlist),loss.float()))
                    s = str(((out.argmax(1)==target.argmax(1)).float().mean()).item() * 100 ) + "%"
                    print("accuracy:{}".format(s))

        torch.save(self.net,"models/net.pth")

if __name__ == '__main__':
    t = Train()
    t.train()