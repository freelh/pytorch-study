import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1),uns函数的作用是添加一个维度
y = x.pow(2) + 0.2*torch.rand(x.size())                 # noisy y data (tensor), shape=(100, 1)

# torch can only train on Variable, so convert them to Variable
x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() #继承  先建立好两层结构，确定形状，类似keras的层
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer  全连接
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.tanh(self.hidden(x))      # activation function for hidden layer   现在添加激活函数
        x = self.predict(x)             # linear output   predict输出即为网络输出
        return x

net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network   ，创建类，确定网络形状
print(net)  # net architecture

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)#创建一个优化器
loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss，创建一个loss计算器

plt.ion()   # something about plotting

for t in range(1000):
    prediction = net(x)     # input x and predict based on x ，根据目前网络参数计算

    loss = loss_func(prediction, y)     # must be (1. nn output, 2. target)，用loss计算器计算loss，注意顺序！
    optimizer.zero_grad()   # clear gradients for next train    清除记忆的残差
    loss.backward()         # backpropagation, compute gradients    BP计算
    optimizer.step()        # apply gradients             将BP计算的反传应用于网络，因为opt传入的是net的参数

    if t % 5 == 0:
        # plot and show learning process
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data[0], fontdict={'size': 20, 'color':  'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()
