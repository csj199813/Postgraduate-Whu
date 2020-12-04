from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np

path = './data'

# 使用Compose 将tensor化和正则化操作打包
transform_fn = Compose([
    ToTensor(),
    Normalize(mean=(0.1307,), std=(0.3081,))
])
mnist_dataset = MNIST(root=path, train=True, transform=transform_fn, download=True)
data_loader = torch.utils.data.DataLoader(dataset=mnist_dataset, batch_size=2, shuffle=True)

# 1. 构建函数,数据集预处理
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
def get_dataloader(train=True, batch_size=BATCH_SIZE):
    '''
    train=True, 获取训练集
    train=False 获取测试集
    '''
    transform_fn = Compose([
        ToTensor(),
        Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = MNIST(root='./data', train=train, transform=transform_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader


class MnistModel(nn.Module):
	def __init__(self):
		super().__init__()  # 继承父类
		self.fc1 = nn.Linear(1 * 28 * 28, 28)  # 添加全连接层
		self.fc2 = nn.Linear(28, 10)

	def forward(self, input):
		x = input.view(-1, 1 * 28 * 28)
		x = self.fc1(x)
		x = F.relu(x)
		out = self.fc2(x)
		return F.log_softmax(out, dim=-1)  # log_softmax 与 nll_loss合用，计算交叉熵

mnist_model = MnistModel()
optimizer = torch.optim.Adam(params=mnist_model.parameters(), lr=0.001)

# 如果有模型则加载
# if os.path.exists('./model'):
# 	mnist_model.load_state_dict(torch.load('model/mnist_model.pkl'))
# 	optimizer.load_state_dict(torch.load('model/optimizer.pkl'))


def train(epoch):
	data_loader = get_dataloader()

	for index, (data, target) in enumerate(data_loader):
		optimizer.zero_grad()  # 梯度先清零
		output = mnist_model(data)
		loss = F.nll_loss(output, target)
		loss.backward()  # 误差反向传播计算
		optimizer.step()  # 更新梯度

		if index % 100 == 0:
			# 保存训练模型
			# torch.save(mnist_model.state_dict(), 'model/mnist_model.pkl')
			# torch.save(optimizer.state_dict(), 'model/optimizer.pkl')
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, index * len(data), len(data_loader.dataset),
					   100. * index / len(data_loader), loss.item()))


def test():
	loss_list = []
	acc_list = []

	test_loader = get_dataloader(train=False, batch_size=TEST_BATCH_SIZE)
	mnist_model.eval()  # 设为评估模式

	for index, (data, target) in enumerate(test_loader):
		with torch.no_grad():
			out = mnist_model(data)
			loss = F.nll_loss(out, target)
			loss_list.append(loss)

			pred = out.data.max(1)[1]
			acc = pred.eq(target).float().mean()  # eq()函数用于将两个tensor中的元素对比，返回布尔值
			acc_list.append(acc)

	print('平均准确率, 平均损失', np.mean(acc_list), np.mean(loss_list))


for i in range(5):
	train(i)

test()