import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import glob


seconds = 10

#name = 'dancer.jpeg'

#image_path = '/Users/orlandoleone/Documents/docs/' + name

#img = cv2.imread(image_path)

model = hub.load('/Users/orlandoleone/Documents/docs/movenet_singlepose_thunder_4')
movenet = model.signatures['serving_default']

h = 1
w = 1

color = (0, 255, 0)

thick = 7

arr = [[16, 14], [14, 12], [6, 8], [8, 10], [15, 13], [13, 11], [5, 7], [7, 9], [11, 12], [5, 6], [2, 2], [1, 1], [0, 0], [11, 5], [12, 6]]

def bigplotter(frame, i):

    image_path = 'docs/extra/frame_' + str(i) + '.jpeg'
    
    img1 = cv2.imread(image_path)

    dimensions = img1.shape

    h = img1.shape[0]
    w = img1.shape[1]
    
    buffer = abs(w-h)

    if h > w :
        img = img1[int(buffer/2):h-int(buffer/2), 0:w]
        h = w
    else :
        img = img1[0:h, int(buffer/2):w-int(buffer/2)]
        w = h

    cv2.imwrite(image_path, img)

    # cv2.imwrite('docs/' + name, img)

    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)

    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    outputs = movenet(image)
    keypoints = outputs['output_0']

    for j in range(15):
        starter = arr[j][0];
        ender = arr[j][1]

        if (keypoints[0][0][starter][2] < 0.25 or keypoints[0][0][ender][2] < 0.25) : continue
        
        start = (int(w*keypoints[0][0][starter][1]), int(h*keypoints[0][0][starter][0]))
        end = (int(w*keypoints[0][0][ender][1]), int(h*keypoints[0][0][ender][0]))
        cv2.line(img, start, end, color, thick)
    

    cv2.imwrite('docs/extra/frame_' + str(i) + '.jpeg', img)

# cv2.imshow('ORLANDO IS THE GOAT', img)

#add line equation function
#calculate angles


video = cv2.VideoCapture('/Users/orlandoleone/Documents/docs/Stephen Curry Teaches Shooting, Ball-Handling, and Scoring | Official Trailer | MasterClass (720p).mp4')


for i in range(seconds*30):
    ret, frame = video.read()
    
    video.set(cv2.CAP_PROP_POS_FRAMES, i)

    cv2.imwrite('docs/extra/frame_' + str(i) + '.jpeg', frame)
    
    bigplotter(frame, i)



'''
frameSize = (w, w)

out = cv2.VideoWriter('OUTPUT.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

for filename in glob.glob('/Users/orlandoleone/Documents/docs/extra/*.jpeg'):
    img = cv2.imread(filename)
    out.write(img)

out.release()
print("done")

'''

'''
cap = cv2.VideoCapture(0)   

while cap.isOpened():
    ret, frame = cap.read()
    print("1")
    

'''

''' indices = tf.constant([[[0, 0], [0, 1]], [[1, 1], [1, 3]]])
 result = tf.gather_nd(keypoints, indices) with tf.Session() as sess:
    print(sess.run(result)) '''
