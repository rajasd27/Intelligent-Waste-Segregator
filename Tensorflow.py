import os
import numpy as np
from random import shuffle
from tqdm import tqdm
import tflearn
import cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from PIL import Image
import dropbox

TRAIN_DIR = '/home/rajas/be_project/training_set'
TRAIN_DIR1 = '/home/rajas/be_project/new_train_set'
TRAIN_DIR2 = '/home/rajas/be_project/google_images'
TRAIN_DIR3 = '/home/rajas/be_project/google_rotated_train'
TRAIN_DIR4 = '/home/rajas/be_project/paper_crumpled_done'
TRAIN_DIR5 = '/home/rajas/be_project/paper_rotated'

def label_img(img):
	word_label = img.split('.')[0]
	if 'metal' in word_label: 
		return np.array([1,0,0,0])
	elif 'glass' in word_label: 
		return np.array([0,0,0,1])
	elif 'plastic' in word_label: 
		return np.array([0,1,0,0])
	elif 'paper' in word_label: 
		return np.array([0,0,1,0])
	elif 'cardboard' in word_label: 
		return np.array([0,0,1,0])


def create_train_data():
	training_data = []
	for img in tqdm(os.listdir(TRAIN_DIR5)):
		#print(img)
		label = label_img(img)
		#print(label)
		path = os.path.join(TRAIN_DIR5,img)
		#print(path)
		img = cv2.imread(path,1)
		img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
		float_img = img.astype(float)
		float_img = float_img.reshape([-1,3,32,32])
		float_img = float_img.transpose([0,2,3,1])
		training_data.append([float_img, label])

	for img in tqdm(os.listdir(TRAIN_DIR4)):
		#print(img)
		label = label_img(img)
		#print(label)
		path = os.path.join(TRAIN_DIR4,img)
		#print(path)
		img = cv2.imread(path,1)
		img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
		float_img = img.astype(float)
		float_img = float_img.reshape([-1,3,32,32])
		float_img = float_img.transpose([0,2,3,1])
		training_data.append([float_img, label])

	for img in tqdm(os.listdir(TRAIN_DIR3)):
		#print(img)
		label = label_img(img)
		#print(label)
		path = os.path.join(TRAIN_DIR3,img)
		#print(path)
		img = cv2.imread(path,1)
		img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
		float_img = img.astype(float)
		float_img = float_img.reshape([-1,3,32,32])
		float_img = float_img.transpose([0,2,3,1])
		training_data.append([float_img, label])

	for img in tqdm(os.listdir(TRAIN_DIR2)):
		#print(img)
		label = label_img(img)
		#print(label)
		path = os.path.join(TRAIN_DIR2,img)
		#print(path)
		img = cv2.imread(path,1)
		img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
		float_img = img.astype(float)
		float_img = float_img.reshape([-1,3,32,32])
		float_img = float_img.transpose([0,2,3,1])
		training_data.append([float_img, label])

	for img in tqdm(os.listdir(TRAIN_DIR1)):
		#print(img)
		label = label_img(img)
		#print(label)
		path = os.path.join(TRAIN_DIR1,img)
		#print(path)
		img = cv2.imread(path,1)
		img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
		float_img = img.astype(float)
		float_img = float_img.reshape([-1,3,32,32])
		float_img = float_img.transpose([0,2,3,1])
		training_data.append([float_img, label])

	for img in tqdm(os.listdir(TRAIN_DIR)):
		#print(img)
		label = label_img(img)
		#print(label)
		path = os.path.join(TRAIN_DIR,img)
		#print(path)
		img = cv2.imread(path,1)
		img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
		float_img = img.astype(float)
		float_img = float_img.reshape([-1,3,32,32])
		float_img = float_img.transpose([0,2,3,1])
		training_data.append([float_img, label])


	shuffle(training_data)
	with open('training_data.pickle','wb') as f:
		pickle.dump(training_data,f)
	
	return training_data

if os.path.exists('training_data.pickle'):
	pickle_in = open('training_data.pickle','rb')
	train_data = pickle.load(pickle_in)
else:
	train_data = create_train_data()


img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=15.)

network = input_data(shape=[None, 32, 32, 3], data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 32, 3, strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = conv_2d(network, 64, 3, strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
network = conv_2d(network, 64, 3 , strides=1, padding='same', activation='relu', bias=True, bias_init='zeros', weights_init='uniform_scaling')
network = max_pool_2d(network, 2 , strides=None, padding='same')
network = fully_connected(network, 512, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam', loss='categorical_crossentropy', learning_rate=0.001)

train = train_data[:-200]
test = train_data[-200:]

X = np.array([i[0] for i in train]).reshape(-1, 32, 32, 3)
Y = np.array([i[1] for i in train])


test_x = np.array([i[0] for i in test]).reshape(-1, 32, 32, 3)
test_y = np.array([i[1] for i in test])


model = tflearn.DNN(network, tensorboard_dir='log')
model.load('quicktest2.model')
#model.fit(X, Y, n_epoch=50, shuffle=True, validation_set= (test_x, test_y) , show_metric=True, batch_size=100 , run_id='aa2')

#model.save('quicktest2.model')
dbx = dropbox.Dropbox('6Xjl6rD5c1QAAAAAAAACL61p6g_XmV7oE3a3nN6SYd64LYw_zAhUbGIW3KZ6xenn')

def download(file_name,target_name):
	with open(target_name, "wb") as f:
		md,res = dbx.files_download(file_name)
		f.write(res.content)

def upload(file_name,path_name):
	meta = dbx.files_upload(file_name.read(),path_name,mode=dropbox.files.WriteMode("overwrite"))

while True:
	for entry in dbx.files_list_folder('/be_proj').entries:
		print(entry.name)
		if 'input.jpg' in entry.name:
			download('/be_proj/input.jpg','image.jpg')
			img = cv2.imread('image.jpg',1)
			img = cv2.resize(img, (32,32), interpolation = cv2.INTER_AREA)
			float_img = img.astype(float)
			float_img = float_img.reshape([-1,3,32,32])
			float_img = float_img.transpose([0,2,3,1])

			t = np.argmax(model.predict(float_img))
			print(t)

			f = open("prediction.txt","w")
			f.write(str(t))
			f.close()

			file_name = open('prediction.txt', 'rb')
			path_name = '/be_proj/prediction.txt'

			upload(file_name,path_name)
			dbx.files_delete('/be_proj/input.jpg')