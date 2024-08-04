import json

class Statistical_sample:
	def	__init__(self) -> None:
		self.error = {
			'happiness': [0] * 15, 
			'surprise': [0] * 15,
			'sadness': [0] * 15,
			'anger': [0] * 15,
			'fear': [0] * 15,
			'disgust': [0] * 15
		}
		self.total = {
			'happiness': [0] * 35, 
			'surprise': [0] * 35,
			'sadness': [0] * 35,
			'anger': [0] * 35,
			'fear': [0] * 35,
			'disgust': [0] * 35
		}
		self.right = {
			'happiness': [0] * 35, 
			'surprise': [0] * 35,
			'sadness': [0] * 35,
			'anger': [0] * 35,
			'fear': [0] * 35,
			'disgust': [0] * 35
		}
	
	def process_sample(self, emotion, positions, right):
		if positions[0] - positions[1] >= 35:
			print('positions[0] - positions[1]', positions[0] - positions[1])
		EMOTION_LIST = {
            'happiness': 0, 
            'surprise': 1,
            'sadness': 2,
            'anger': 3,
            'fear': 4,
            'disgust': 5
        }

		if emotion == "happines":
			emotion = "happiness"
		if emotion == 'excited':
			emotion = "happiness"
		if emotion == 'sad':
			emotion = "sadness"
		if emotion == 'surprised':
			emotion = 'surprise'
		if emotion == 'happy':
			emotion = 'happiness'
		if emotion == 'angry':
			emotion = 'anger'


		if EMOTION_LIST[emotion] == 0:
			self.total['happiness'][positions[0] - positions[1]] += 1
		elif EMOTION_LIST[emotion] == 1:
			self.total['surprise'][positions[0] - positions[1]] += 1
		elif EMOTION_LIST[emotion] == 2:
			self.total['sadness'][positions[0] - positions[1]] += 1
		elif EMOTION_LIST[emotion] == 3:
			self.total['anger'][positions[0] - positions[1]] += 1
		elif EMOTION_LIST[emotion] == 4:
			self.total['fear'][positions[0] - positions[1]] += 1
		else:
			self.total['disgust'][positions[0] - positions[1]] += 1
		if right:
			if EMOTION_LIST[emotion] == 0:
				self.right['happiness'][positions[0] - positions[1]] += 1
			elif EMOTION_LIST[emotion] == 1:
				self.right['surprise'][positions[0] - positions[1]] += 1
			elif EMOTION_LIST[emotion] == 2:
				self.right['sadness'][positions[0] - positions[1]] += 1
			elif EMOTION_LIST[emotion] == 3:
				self.right['anger'][positions[0] - positions[1]] += 1
			elif EMOTION_LIST[emotion] == 4:
				self.right['fear'][positions[0] - positions[1]] += 1
			else:
				self.right['disgust'][positions[0] - positions[1]] += 1

	
	def writeFile(self, pred_file):
		pred_file.write('eval' + '\n')
		for key, value in self.error.items():
			pred_file.write(key + ': ' + str(value) + '\n')
		
		pred_file.write('total' + '\n')
		for key, value in self.total.items():
			pred_file.write(key + ': ' + str(value) + '\n')

		pred_file.write('right' + '\n')
		for key, value in self.right.items():
			pred_file.write(key + ': ' + str(value) + '\n')
	


if __name__ == '__main__':
	data_dir = 'data/original_annotation/'
	file = [
		'dailydialog_test',
		'dailydialog_train'
	]
	pred_file = open('statistical_sample_size.txt', 'w')
	statistical_sample = Statistical_sample()
	for file_name in file:
		data_file = data_dir + file_name + '.json'  
		f = open(data_file, 'r')
		content = f.read()
		dataset = json.loads(content)
		for num, tr in dataset.items():
			for i in range(len(tr[0])):
				if tr[0][i]['emotion'] != 'neutral':
					pair = tr[0][i]['expanded emotion cause evidence']
					for j in range(i+1):
						if j+1 in pair:
							print('[i, num - 1]', [i, j])
							statistical_sample.process_sample(tr[0][i]['emotion'], [i, j], right=True)
						else:
							statistical_sample.process_sample(tr[0][i]['emotion'], [i, j], right=False)
	statistical_sample.writeFile(pred_file)


		