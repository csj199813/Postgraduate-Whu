from my_dis import distance
import random
import math
def sa(path, dis_map, things_num):
	init_temp = 120
	k = 0.95
	lowest = 0.01
	iter = 300
	M = 250
	count = 0
	current_temp = init_temp
	lest_length = distance(dis_map, path, things_num)
	while(current_temp > lowest and M > count):
		count = 0
		for i in range(iter):
			m = random.sample(list(range(len(path))), 2)
			new_path = path[:]
			new_path[m[0]], new_path[m[1]] = new_path[m[1]], new_path[m[0]]
			length = distance(dis_map, new_path, things_num)
			delta = length - lest_length
			rand = random.random()
			p = math.exp(-delta/current_temp)
			if delta < 0:
				path = new_path
				lest_length = length
			elif rand < p:
				path = new_path
				lest_length = length
			else:
				count += 1
		current_temp *= k
	return path, lest_length


