#author   BKL
#contact  804872510@qq.com
#last update  2022.6.17

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable


x = torch.unsqueeze(torch.linspace(-1,1,100),dim=1)     #create data for test
y = x.pow(3)+0.1*torch.randn(x.size())
x , y =(Variable(x),Variable(y))   #trans data for pytorch variable
#plt.scatter(x.data,y.data)
#plt.show()

class Net(nn.Module):
    def __init__(self,n_input,n_hidden,n_output):
        super(Net,self).__init__()
        self.hidden1 = nn.Linear(n_input,n_hidden)      #nn.linear is set full connect layer
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)
    def forward(self,input):      #forward propa
        out = self.hidden1(input)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out =self.predict(out)
        return out


net = Net(1,50,1)
print(net)

optimizer = torch.optim.SGD(net.parameters(),lr = 0.01)
loss_func = torch.nn.MSELoss()

plt.ion()
plt.show()

for t in range(2000):
    prediction = net(x)
    loss = loss_func(prediction,y)
    loss.cuda()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if t%5 ==0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'epoch:%s   Loss = %.4f' %( str(t),loss.data ), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.05)
    if(t == 1999):
        plt.text(0,1,"train end",fontdict={'size': 20, 'color': 'green'})

plt.ioff()
plt.show()