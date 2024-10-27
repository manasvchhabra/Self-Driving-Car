import os
import cv2
import numpy as np
import tensorflow as tf
import sys
import serial
import time

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
tf.keras.backend.set_session(tf.Session(config=config));

from utils import label_map_util
from utils import visualization_utils as vis_util

#ser = serial.Serial('/dev/ttyACM0', 115200, timeout=None);

MODEL_NAME = 'finalModel'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'obdet.pbtxt')

NUM_CLASSES = 5

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
		label_map, max_num_classes = NUM_CLASSES, use_display_name = True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name ='')

	sess = tf.Session(graph = detection_graph)

# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('https://192.168.43.1:8080/video')
#Counters for detection
person_det = False
stop_det = False
speed5_det = False
speed10_det = False
person_count = 0
stop_count = 0
speed5_count = 0
speed10_count = 0
count_thres = 20
max_count_thres = count_thres+2
red_rate = 0.006
speed = 7
speed_before = 0
ard_count = 0
while True:
	if speed != 0:
		speed_before = speed
	# Load image using OpenCV and
	# expand image dimensions to have shape: [1, None, None, 3]
	ret, image = cap.read()
	# i.e. a single-column array, where each item in the column has the pixel RGB value
	image_expanded = np.expand_dims(image, axis = 0)
	#Defining some variables for object detection
	mst = 0.7
	#For 640x480 res: 720.6857 focus
	#Pixel width/height = P, D = Actual Distance, M = Actual height/width, then F = P*D/M (F is constant because camera uses pinhole principle. (In pixels)
	focus = 635.6857
	stop_sign_height = 3.3 #In cm
	person_height = 6
	person2_height = 5.2
	speed_limit_5_height = 8.2
	speed_limit_10_height = 7.3
	frame_height, frame_width = np.shape(image)[0], np.shape(image)[1]

	# Perform the actual detection by running the model with the image as input
	(boxes, scores, classes, num) = sess.run(
		[detection_boxes, detection_scores, detection_classes, num_detections],
		feed_dict ={image_tensor: image_expanded})
	# Draw the results of the detection (aka 'visualize the results')
	vis_util.visualize_boxes_and_labels_on_image_array(
		image,
		np.squeeze(boxes),
		np.squeeze(classes).astype(np.int32),
		np.squeeze(scores),
		category_index,
		use_normalized_coordinates = True,
		line_thickness = 8,
		min_score_thresh = mst)
	for i, b in enumerate(boxes[0]):
			if scores[0][i] > mst:
				#Multiplied by frame dimensions because the coordinates are normalized
				x1 = boxes[0][i][1]*frame_width #Top left point or bottom left point (1st with left to right - hence 1)
				y2 = boxes[0][i][2]*frame_height #Bottom left or bottom right (2nd with top to bottom - hence 2)
				y1 = boxes[0][i][0]*frame_height #Top left or top right (1st with top to bottom - hence 1)

				#For distance calculation:
				#Imagine a pinhole camera, with the projection coordinates (inside the pinhole/camera) (length of x-y, focus, etc.) scaled up say, to the point where 1 px = 1 cm
				#(or basically when the dimension of the object matches the dimension of the coordinate of projection.
				#Then, by similarity, the ratio of pixel height (or other dim) and focal length (which can be experimentally calculated) = the ratio of actual height (or other dim) 					#of object and the distance to the object from the pinhole.
				#The focal length here is the product of actual camera focus and an upscale factor (relating to the ratio of size of image and the capture screen)
				#or just think of the ratio of the pixel length and capture screen to be constant since the display of the image can be altered - hence 1px = 1cm

				if classes[0][i] == 1:
					distance_person = focus*(person_height+person2_height)/2/(y2-y1)
					person_count += 2
				elif classes[0][i] == 2:
					distance_person = focus*(person_height+person2_height)/2/(y2-y1)
					person_count += 2
				elif classes[0][i] == 3:
					distance_stop = focus*stop_sign_height/(y2-y1)
					stop_count +=1
				elif classes[0][i] == 4:
					distance_speed5 = speed_limit_5_height*focus/(y2-y1)
					speed5_count +=1
				elif classes[0][i] == 5:
					distance_speed10 = speed_limit_10_height*focus/(y2-y1)
					speed10_count +=1

				if person_count >= max_count_thres:
					person_count = count_thres
				if stop_count >= max_count_thres:
					stop_count = count_thres
				if speed5_count >= max_count_thres:
					speed5_count = count_thres
				if speed10_count >= max_count_thres:
					speed10_count = count_thres

				if stop_count >= count_thres:
					pass
				if speed5_count >= count_thres:
					speed = 5
				if speed10_count >= count_thres:
					speed = 10
				if person_count >= count_thres-5:
					if distance_person < 15:
						speed = 0
			else:
					#ser.write("00-"+str(speed))
					#cv2.putText(image, str(round(distance, 2)), (int(x1)+10, int(y2)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
					#if distance < 20: #If distance is less than 20cm
						#cv2.putText(image, 'STOP', (int(x1)+10, int(y1)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
				if person_count > 0:
					person_count -= red_rate
				if stop_count > 0:
					stop_count -= red_rate
				if speed5_count > 0:
					speed5_count -= red_rate
				if speed10_count > 0:
					speed10_count -= red_rate
				if person_count <= 0 and speed5_count <= 0 and speed10_count <= 0:
					speed = speed_before

			print("speed 5 count", speed5_count)
			print("speed 10 count", speed10_count)
			print("person count", person_count)
			print("speed", speed)
			transByte = str("00") + "-" + str(speed) + "\r\n"
			if ard_count == 3:
				#ser.write(b'HHHHH')
				ard_count = 0
			ard_count += 1
			print("end")
	cv2.imshow('capture', image)
	#cap.release()
	if cv2.waitKey(50) == ord('q'):
		cv2.destroyAllWindows()
		break
