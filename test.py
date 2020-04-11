import os
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import cv2
from scipy import ndimage


def box(matrix):
	labeled_image, num_features = ndimage.label(matrix)
	# Find the location of all objects
	objs = ndimage.find_objects(labeled_image)

	boxes = []
	for ob in objs:
		boxes.append([int(ob[0].start), int(ob[0].stop), int(ob[1].start), int(ob[1].stop)])
		
	return(boxes)
	

def boxNear(matrix):
	
	def explore(i,j):
		if matrix[i][j]==0:
			return [[i],[j]]
		
		matrix[i][j]=0
		coords = [[i],[j]]
		if i>=1:
			res1 = explore(i-1,j)
			coords[0]+=res1[0]
			coords[1]+=res1[1]
		if i<matrix.shape[0]-1:
			res2 = explore(i+1,j)
			coords[0]+=res2[0]
			coords[1]+=res2[1]
		if j>=1:
			res3 = explore(i,j-1)
			coords[0]+=res3[0]
			coords[1]+=res3[1]
		if j<matrix.shape[1]-1:
			res4 = explore(i,j+1)
			coords[0]+=res4[0]
			coords[1]+=res4[1]
		return coords
		
	boxes = []
	for i in range(matrix.shape[0]):
		for j in range(matrix.shape[1]):
			if matrix[i][j]==1:
				coords = explore(i,j)
				boxes.append([min(coords[0]), min(coords[1]), max(coords[0]), max(coords[1])])
	return boxes
	

	
def template(path='template.jpg'):
	data_path = path
	I = Image.open(data_path)
	I = np.asarray(I)
	red = np.array([[I[i,j,0]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
	green = np.array([[I[i,j,1]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
	blue = np.array([[I[i,j,2]*2/255.0-1 for j in range(I.shape[1])]for i in range(I.shape[0])])
	
	return red, green, blue


def detect_red_light(I):
	'''
	Note that PIL loads images in RGB order, so:
	I[:,:,0] is the red channel
	I[:,:,1] is the green channel
	I[:,:,2] is the blue channel
	'''
	
	
	bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
	
	'''
	BEGIN YOUR CODE
	'''
	
	
	temp_all_channel = template(path='template.jpg')
	temp_red = temp_all_channel[0]  #use red channel
	ground_truth = np.sum(temp_red*temp_red)
	h = temp_red.shape[0]
	w = temp_red.shape[1]
	print(h,w)
	red_ch = I[:,:,0]
	conv_res = [[0]*(red_ch.shape[1]-h) for _ in range(red_ch.shape[0]-w)]
	for i in range(red_ch.shape[0]-h):
		for j in range(red_ch.shape[1]-w):
			test_area = red_ch[i:i+h, j:j+w]
			#print(i,j,test_area.shape)
			#print(test_area)
			conv_res[i][j]=abs(np.sum(temp_red*test_area)-ground_truth)
			#print(np.sum(temp_red*test_area))
	conv_res = conv_res/np.max(conv_res)
	conv_res = np.where(conv_res<0.01,1,0) 
	print(conv_res) 
	ax = sns.heatmap(np.array(conv_res))
	#ax = sns.heatmap(np.array(red_ch))
	#plt.show()
	
	boxes_from_red = boxNear(conv_res)	
	#boxes_from_red = box(conv_res)	

	return boxes_from_red
	
	'''
	temp_green = temp_all_channel[1]
	temp_g_avg = np.mean(temp_green)
	temp_blue = temp_all_channel[2]
	temp_b_avg = np.mean(temp_blue)
	
	print("temp_green_avg",temp_g_avg)
	print("temp_blue_avg",temp_b_avg)
	
	boxes_final = []
	for a,b,c,d in boxes_from_red:
		test_area_r_avg = np.mean(I[a:c+1,b:d+1,0]*2/255.0-1)
		test_area_g_avg = np.mean(I[a:c+1,b:d+1,1]*2/255.0-1)
		test_area_b_avg = np.mean(I[a:c+1,b:d+1,2]*2/255.0-1)
		
		print(test_area_g_avg, test_area_b_avg)
		if abs(test_area_b_avg-temp_b_avg)<0.5 and abs(test_area_g_avg-temp_g_avg)<0.5:
			boxes_final.append([a,b,c,d])
	'''
	

		


pic_I = Image.open('../RedLights2011_Medium/RL-131.jpg')
pic_I = np.asarray(pic_I)
boxes = detect_red_light(pic_I)
print(boxes)
