import cv2
import os
import time
import uuid
IMAGES_PATH = r'D:\sign_detection\RealTimeObjectDetection\Tensorflow\workspace\images'
labels = ['hello', 'thanks', 'yes', 'no', 'iloveyou']
num_imgs = 15
for label in labels:
  os.mkdir(r'E:\sign_detection\RealTimeObjectDetection\Tensorflow\workspace\images\\'+label)
  cap = cv2.VideoCapture(0)
  print('Collecting images for {}'.format(label))
  time.sleep(5)
  for imgnum in range(num_imgs):
    ret, frame = cap.read()
    imagename = os.path.join(IMAGES_PATH, label, label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
    cv2.imwrite(imagename, frame)
    cv2.imshow('frame', frame)
    time.sleep(2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  cap.release()