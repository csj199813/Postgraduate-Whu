import numpy as np
import os
from collections import OrderedDict

train_path = 'digits/trainingDigits'
test_path = 'digits/testDigits'

#激活函数
def relu(x):
	return np.maximum(0, x)

#softmax函数，多分类
def softmax(x):
	max = np.max(x)
	a = np.exp(x - max)
	sum = np.sum(a, axis=1, keepdims=True)
	y = a / sum
	return y

#交叉熵函数计算loss
def cross_entropy(y, t):
	if y.ndim == 1:
		y = y.reshape(1, -1)
		t = t.reshape(1, -1)
	batch_size = y.shape[0]
	delta = 1e-7
	return -np.sum(t * np.log(y + delta))/batch_size
#全连接网络
class Fully_Connect_Net():
	#初始化参数
	def __init__(self, input_size, hidden_size, output_size, lr=0.01):
		self.lr = lr
		self.params = {}
		# np.random.seed(7)
		self.params['w1'] = np.sqrt(2 / input_size) * np.random.randn(input_size, hidden_size)
		self.params['b1'] = np.zeros(hidden_size)
		self.params['w2'] = np.sqrt(2 / hidden_size) * np.random.randn(hidden_size, output_size)
		self.params['b2'] = np.zeros(output_size)

	#向前传播
	def forward(self, x):
		self.x = x
		w1, w2 = self.params['w1'], self.params['w2']
		b1, b2 = self.params['b1'], self.params['b2']
		self.a1 = np.dot(self.x, w1) + b1
		self.z1 = relu(self.a1)
		self.a2 = np.dot(self.z1, w2) + b2
		self.z2 = relu(self.a2)
		self.y = softmax(self.z2)
		return self.y
	#向后传播
	def backward(self, t):
		batch_size = self.y.shape[0]
		soft_grad = (self.y - t)
		a2 = np.multiply(soft_grad, (self.a2 > 0))
		w2_grad = np.dot(self.z1.T, a2) / batch_size
		b2_grad = np.sum(a2, axis=0) / batch_size
		dout_1 = np.dot(a2, self.params['w2'].T)
		a1 = np.multiply(dout_1, (self.a1 > 0))
		w1_grad = np.dot(self.x.T, a1) / batch_size
		b1_grad = np.sum(a1, axis=0) / batch_size
		# print('w1, %s w2,%s b1,%s b2 %s'%(w1_grad, w2_grad, b1_grad, b2_grad))
		#更新参数
		self.params['w1'] -= self.lr * w1_grad
		self.params['b1'] -= self.lr * b1_grad
		self.params['w2'] -= self.lr * w2_grad
		self.params['b2'] -= self.lr * b2_grad
	#预测
	def predict(self, x):
		y = self.forward(x)
		index = np.argmax(y)
		return index

label_list = [] #保存训练label
matrix = [] #保存训练样本向量
train_len = len(os.listdir('digits/trainingDigits')) #训练集个数
test_len = len(os.listdir('digits/testDigits')) #测试集个数
#读取训练样本文件，并转换成(样本数，1024列)
for file in os.listdir(train_path):
	for i in range(10):
		if i == int(file[0]):
			label_list.append(1)
		else:
			label_list.append(0)

	file = os.path.join(train_path, file)
	with open(file, 'r') as f:
		f1 = f.readlines()
		for i in f1:
			for j in range(len(i) - 1):
				matrix.append(int(i[j]))
matrix = np.array(matrix).reshape(-1, 1024)
label_list = np.array(label_list).reshape(-1, 10)
#定义并且训练网络
net = Fully_Connect_Net(input_size=1024, hidden_size=2000, output_size=10, lr=0.0001)
mini_batch = 64 #批处理
epoch = 50
for n in range(epoch):
	for i in range(0, train_len, mini_batch):
		train_data = matrix[i:min(i + mini_batch, train_len - 1)]
		train_label = label_list[i:min(i + mini_batch, train_len - 1)]
		y = net.forward(train_data)
		loss = cross_entropy(y, train_label)
		net.backward(train_label)
		print('第{}次训练 loss：{:.4f}'.format(n, loss))

#在测试集上进行测试
right = 0  # 保存正确预测个数
error = 0
for num, file in enumerate(os.listdir(test_path)):
	print('正在预测第%s个样本' % num, file)
	label = file[0]
	file = os.path.join(test_path, file)
	test = []
	with open(file, 'r') as f:
		f1 = f.readlines()
		for i in f1:
			for j in range(len(i) - 1):
				test.append(int(i[j]))
	test = np.array(test).reshape(1, -1)  # 单个测试样本向量
	predict = net.predict(test)
	print(predict, label)
	if str(predict) == label:
		right += 1
	else:
		error += 1
print('已经测试完毕 精确度为：{:.4f}'.format(right/test_len))
print('已经测试完毕 错误个数：{:.4f}'.format(error))



