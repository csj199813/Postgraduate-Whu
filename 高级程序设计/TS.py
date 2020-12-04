from my_dis import distance
#搜索邻居节点,返回邻居列表和操作列表
def search_neighbor(city_path):
	path_list = []
	exchange_list = []
	for i in range(len(city_path) - 1):
		for j in range(i + 1, len(city_path)):
			new_path = city_path[:]
			exchange_list.append([new_path[i], new_path[j]])
			new_path[i], new_path[j] = new_path[j], new_path[i]
			path_list.append(new_path)
	return path_list, exchange_list
#计算每个邻居所花费的距离
def cal_dis_list(path_list, dis_map):
	dis_list = []
	for i in path_list:
		dis = distance(dis_map, i, len(i))
		dis_list.append(dis)
	return dis_list

#表长不易过短或过长，过短会导致局部最优解，过长计算时间太长
def ts(city_path, dis_map, table_length):
	#初始化参数
	path_list = []
	path_list.append(city_path)
	dis_list = cal_dis_list(path_list, dis_map) #每条路径的花费列表
	lest_dis = min(dis_list) #最短距离
	lest_path = path_list[dis_list.index(lest_dis)] #最短路径
	init_path = lest_path #初始路径
	ban_table = [] #禁忌表
	iter = 500 #迭代次数

	for i in range(iter):
		neighbor_path, exchange_list = search_neighbor(init_path)
		dis_list = cal_dis_list(neighbor_path, dis_map)
		current_dis = min(dis_list)
		#如果当前距离小于最短距离，直接取当前距离，破除禁忌表
		if current_dis < lest_dis:
			lest_dis = current_dis
			lest_path = neighbor_path[dis_list.index(current_dis)]
			init_path = neighbor_path[dis_list.index(current_dis)]
			if exchange_list[dis_list.index(current_dis)] in ban_table:
				ban_table.remove(exchange_list[dis_list.index(current_dis)]) #破禁
				ban_table.append(exchange_list[dis_list.index(current_dis)]) #加入禁忌表
			else:
				ban_table.append(exchange_list[dis_list.index(current_dis)])
		#如果当前距离大于最短距离，并且该操作在禁忌表中，则寻找一个次优解（不在禁忌表中的解）
		elif exchange_list[dis_list.index(current_dis)] in ban_table:
			while exchange_list[dis_list.index(current_dis)] in ban_table:
				neighbor_path.remove(neighbor_path[dis_list.index(current_dis)]) #remove代表移除列表中该值操作，index代表取出该值的列表下标
				exchange_list.remove(exchange_list[dis_list.index(current_dis)])
				dis_list.remove(dis_list[dis_list.index(current_dis)])
				current_dis = min(dis_list)
			init_path = neighbor_path[dis_list.index(current_dis)]
			ban_table.append(exchange_list[dis_list.index(current_dis)])
		#如果当前距离大于最短距离且不在禁忌表中，则选择该解
		else:
			init_path = neighbor_path[dis_list.index(current_dis)]
			ban_table.append(exchange_list[dis_list.index(current_dis)])
		if len(ban_table) > table_length:
			ban_table.remove(ban_table[0])

	return lest_path, lest_dis





