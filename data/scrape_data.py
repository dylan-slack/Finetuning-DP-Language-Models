import sqlite3
import pandas as pd

from copy import deepcopy

for data_set_size in [5000, 10000, 20000]:

	sql_conn = sqlite3.connect('data/reddit-comments-may-2015/database.sqlite')
	df = pd.read_sql("SELECT score, body FROM May2015 WHERE LENGTH(body) > 150 AND LENGTH(body) < 1000 LIMIT 40000", sql_conn)

	arr = df.to_numpy()

	with open('data/reddit-{}.train.txt'.format(data_set_size),'a') as f:
		start = 10000
		i = deepcopy(start)
		while i < start + data_set_size:
			next_comment = arr[i, 1]
			f.write(next_comment.replace("\n", "") + "\n")
			i += 1

	with open('data/reddit-{}.dev.txt'.format(data_set_size),'a') as f:
		for i in range(0,5000,1):
			next_comment = arr[i, 1]
			f.write(next_comment.replace("\n", "") + "\n")

	with open('data/reddit-{}.test.txt'.format(data_set_size),'a') as f:
		for i in range(5000,10000,1):
			next_comment = arr[i, 1]
			f.write(next_comment.replace("\n", "") + "\n")



