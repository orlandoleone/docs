import cv2
import tensorflow as tf
import tensorflow_hub as hub

name = 'dancer.jpeg'

image_path = '/Users/orlandoleone/Documents/docs/' + name

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
    

cv2.imwrite('docs/' + name, img)

image = tf.io.read_file(image_path)
image = tf.compat.v1.image.decode_jpeg(image)
image = tf.expand_dims(image, axis=0)

image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)


model = hub.load('/Users/orlandoleone/Documents/docs/movenet_singlepose_thunder_4')
movenet = model.signatures['serving_default']

outputs = movenet(image)
keypoints = outputs['output_0']

print(keypoints)

print(keypoints[0][0][16][0])
print(keypoints[0][0][14][0])

color = (0, 255, 0)

thick = 7


def plot(starter, ender):
    start = (int(w*keypoints[0][0][starter][1]), int(h*keypoints[0][0][starter][0]))
    end = (int(w*keypoints[0][0][ender][1]), int(h*keypoints[0][0][ender][0]))
    cv2.line(img, start, end, color, thick)


#right
plot(16, 14)
plot(14, 12)
plot(12, 6)
plot(6, 8)
plot(8, 10)

#left
plot(15, 13)
plot(13, 11)
plot(11, 5)
plot(5, 7)
plot(7, 9)

#hips
plot(11, 12)

#shoulders
plot(5, 6)

#head
plot(1, 1)
plot(2, 2)
plot(0, 0)


#add line equation function
#calculate angles



cv2.imshow('ORLANDO IS THE GOAT', img)



""" indices = tf.constant([[[0, 0], [0, 1]], [[1, 1], [1, 3]]])
 result = tf.gather_nd(keypoints, indices) with tf.Session() as sess:
    print(sess.run(result)) """
