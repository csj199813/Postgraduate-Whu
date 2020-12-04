import math
def dis_map(points, n):
	dis_each = []
	for i in range(n):
		dis_row = []
		for j in range(n):
			dis = math.sqrt(math.pow((points[i][0] - points[j][0]), 2) + math.pow((points[i][1] - points[j][1]), 2))
			dis_row.append(dis)
		dis_each.append(dis_row)
	return dis_each

def distance(map, points, things_num):
	length = 0
	for i in range(things_num - 1):
		length += map[points[i]][points[i + 1]]
	length += map[points[things_num - 1]][points[0]]
	return length