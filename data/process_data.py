import json
import csv

def process_text_with_prompt(emotion, target_utterance, candidate_utterance, history, pair, itself, subone, base):

	prompt = {
		'surprise': " is because of the <mask> utterance ", 
		'intra': " is<mask> because of the utterance itself"
		# 'intra': " is <mask> the utterance itself"
	}
	# if base:
	# 	# text = '" ' + history + ' " </s> ' + emotion + ' </s> " ' + pair[0] + ': ' + target_utterance + ' " </s> " ' + \
	# 	# 	pair[1] + ': ' + candidate_utterance + ' " ' + '<mask>'
	# 	text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + ' , is the following utterance the cause of the above emotional utterance ? " ' + \
	# 		pair[1] + ': ' + candidate_utterance + ' " ' + '<mask>'
	# elif itself:
	# 	text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + \
	# 		prompt['intra'] + ' . '
	# 	# print(text)
	# elif emotion == 'surprise'and subone:
	# 	text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + \
	# 		prompt['surprise'] + ' . '
	# else:
	# 	text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + ' , is the following utterance the cause of the above emotional utterance ? " ' + \
	# 		pair[1] + ': ' + candidate_utterance + ' " ' + '<mask>'


	if base:
		# text = '" ' + history + ' " </s> ' + emotion + ' </s> " ' + pair[0] + ': ' + target_utterance + ' " </s> " ' + \
		# 	pair[1] + ': ' + candidate_utterance + ' " ' + '<mask>'
		text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + ' , is the following utterance the cause of the above emotional utterance ? " ' + \
			pair[1] + ': ' + candidate_utterance + ' " ' + '<mask>'
	elif itself:
		text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + \
			prompt['intra'] + ' . '
		# print(text)
	elif emotion == 'surprise'and subone:
		text = '" ' + history + ' " ' + pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + \
			prompt['surprise'] + ' . '
	elif emotion == 'happiness':
		text = '" ' + history + ' " ' + pair[1] + ' says " ' + candidate_utterance + ' " , <mask> ' + \
			pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + ' .'
	else:
		text = '" ' + history + ' " ' + pair[1] + ' says " ' + candidate_utterance + ' " , <mask> ' + \
			pair[0] + ' says " ' + target_utterance + ' " with ' + emotion + ' .'


	return text


if __name__ == '__main__':
	data_dir = 'data/original_annotation/'
	file = [
		'dailydialog_test',
		'dailydialog_train',
		'dailydialog_valid',
	]

	for file_name in file:
		data_file = data_dir + file_name + '.json'  
		f = open(data_file, 'r')
		content = f.read()
		dataset = json.loads(content)
		# 1. 创建文件对象（指定文件名，模式，编码方式）a模式 为 下次写入在这次的下一行
		save_file_name = "data/processed_data/" + file_name + '.csv'
		with open(save_file_name, "w", encoding="utf-8", newline="") as save_file:
			# 2. 基于文件对象构建 csv写入对象
			csv_writer = csv.writer(save_file, delimiter='\t', quoting=csv.QUOTE_NONE, quotechar=None, escapechar="|")
			# 3. 构建列表头
			name=['', 'emotion', 'text', 'labels', 'position', 'id']
			csv_writer.writerow(name)
			number = 0
			for num, tr in dataset.items():
				history = ''
				for i in range(len(tr[0])):
					history += tr[0][i]['speaker'] + ': ' + tr[0][i]['utterance']
					if tr[0][i]['emotion'] == "happines":
						tr[0][i]['emotion'] = "happiness"
					if tr[0][i]['emotion'] == 'excited':
						tr[0][i]['emotion'] = "happiness"
					if tr[0][i]['emotion'] == 'sad':
						tr[0][i]['emotion'] = "sadness"
					if tr[0][i]['emotion'] == 'surprised':
						tr[0][i]['emotion'] = 'surprise'
					if tr[0][i]['emotion'] == 'happy':
						tr[0][i]['emotion'] = 'happiness'
					if tr[0][i]['emotion'] == 'angry':
						tr[0][i]['emotion'] = 'anger'
					if tr[0][i]['emotion'] != 'neutral':
						cause_num = tr[0][i]['expanded emotion cause evidence']
						for j in range(i+1):
							pair = [tr[0][i]['speaker'], tr[0][j]['speaker']]
							text = []
							if i==j:
								itself = True
							else:
								itself = False
							if i-j == 1:
								subone = True
							else:
								subone = False
							

							text.append(process_text_with_prompt(tr[0][i]['emotion'], tr[0][i]['utterance'], tr[0][j]['utterance'], history, pair, itself, subone, base=False))
							text.append(process_text_with_prompt(tr[0][i]['emotion'], tr[0][i]['utterance'], tr[0][j]['utterance'], history, pair, itself, subone, base=True))
							# print('text', text)
							labels = 1 if j+1 in tr[0][i]['expanded emotion cause evidence'] else 0
							id = num
							position = [i, j]
							# 4. 写入csv文件内容
							z = [
								number,
								tr[0][i]['emotion'],
								text,
								labels,
								position,
								id
							]
							number += 1
							csv_writer.writerow(z)
							if cause_num.count(j+1) > 1:
								for k in range(cause_num.count(j+1)-1):
									pair = [tr[0][i]['speaker'], tr[0][j]['speaker']]
									text = []
									if i==j:
										itself = True
									else:
										itself = False
									if i-j == 1:
										subone = True
									else:
										subone = False

									text.append(process_text_with_prompt(tr[0][i]['emotion'], tr[0][i]['utterance'], tr[0][j]['utterance'], history, pair, itself, subone, base=False))
									text.append(process_text_with_prompt(tr[0][i]['emotion'], tr[0][i]['utterance'], tr[0][j]['utterance'], history, pair, itself, subone, base=True))
									# print('text', text)
									labels = 1 if j+1 in tr[0][i]['expanded emotion cause evidence'] else 0
									id = num
									position = [i, j]
									# 4. 写入csv文件内容
									z = [
										number,
										tr[0][i]['emotion'],
										text,
										labels,
										position,
										id
									]
									number += 1
									csv_writer.writerow(z)
			print(file_name, "写入数据成功")
			# 5. 关闭文件
			save_file.close()

		print(type(dataset))
		f.close()