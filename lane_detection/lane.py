import cv2
import numpy as np
import serial
import time
import serial                                                              
import time                                                                 
'''ArduinoUnoSerial = serial.Serial('com6',9600)  
while True:         
        var = input()                                                      
        if (var == '1'):                                                      
            ArduinoUnoSerial.write('1')                           
        if (var == '0'): 0         
            ArduinoUnoSerial.write('0')               
        if (var == 'fine and you'):       
            ArduinoUnoSerial.write('0')   
 '''              
cap = cv2.VideoCapture('http://192.168.43.225:8080/video')


def det_edges(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    dpi = 10 
    
    lower_black = np.array([0,0,0])
    upper_black = np.array([180,255,80]) 
    
    mask = cv2.inRange(hsv, lower_black, upper_black)
    edges = cv2.Canny(mask, 200, 400)

    return edges

#IT WAS NECESSARY  
def region_of_interest(image):
    height, width = image.shape
    mask = np.zeros_like(image)
    polygon = np.array([[
        (0, height*1/2),
        (width, height*1/2),
        (width, height),
        (0,height)
    ]], np.int32)
    cv2.fillConvexPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(image, mask)

    return cropped_edges


def det_line_segments(image):
    rho = 1 
    angle = np.pi / 180 
    min_threshold = 10 
    line_segments = cv2.HoughLinesP(image, rho, angle, min_threshold, np.array([]), minLineLength=8, maxLineGap=8) 
    return line_segments

#SAMSHA
def make_points(image, line):
    height, width, color = image.shape
    (slope, intercept) = line
    y1 = height
    y2 = int(y1*1/2) 

    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(image, line_segments):
    """
    Combines left and right line segments into a group of left and right lane line, using their slopes and intercepts
    """
    lane_lines = []
    if line_segments is None:
        return lane_lines
    height, width, color = image.shape
    left_fit = []
    right_fit = []
    boundary = 1/3
    left_region_boundary = width*(1-boundary)
    right_region_boundary = width*(boundary)

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue #IGNORING POLES
            fit = np.polyfit((x1,x2), (y1,y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope,intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope,intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(image, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)   
    if len(right_fit) > 0:
        lane_lines.append(make_points(image, right_fit_average))    
    return lane_lines
    
def display_lane_lines(image, lines, line_color, line_width):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)  
    return line_image

def line_to_follow(image, lines):
    if len(lines) == 2: 
        average_line = [( int((lines[0][0][0]+lines[1][0][0])/2), int((lines[0][0][1]+lines[1][0][1])/2) ), ( int((lines[0][0][2]+lines[1][0][2])/2), int((lines[0][0][3]+lines[1][0][3])/2 ))] 
        return average_line
    elif len(lines) == 1:#TO TURN LEFT OR RIGHT TO BE IN THE CENTER OF LANE
        pass
def display_line_to_follow(image, lines, line_color, line_width):
    line_image = np.zeros_like(image)
    if lines is not None:
        cv2.line(line_image, lines[0], lines[1], line_color, line_width)    
    line_image = cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return line_image

while True:
    ret, frame=cap.read()
    edges = det_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segs = det_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segs)
    line_color = (0,255,0)
    line_width = 8
    lanes = display_lane_lines(frame, lane_lines, line_color, line_width)
    line2f = line_to_follow(lanes, lane_lines)
    dline2f = display_line_to_follow(lanes, line2f, line_color, line_width)
    cv2.imshow("line", dline2f)
    if cv2.waitKey(2) == ord('q'):
        cv2.destroyAllWindows()
        break

