# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:30:56 2018

@author: Danling
"""

# -*- coding: utf-8 -*-  
import os, cv2, socketio, base64, shutil, eventlet.wsgi  
import numpy as np  
from keras.models import load_model  
from flask import Flask  
from PIL import Image  
from io import BytesIO  
from datetime import datetime  
from keras.preprocessing.image import array_to_img  
  
# socketio  
sio = socketio.Server()  
  
# ------图像预处理-------------  
  
# 除去顶部的天空和底部的汽车正面  
def crop(image):  
    return image[60:-25, :, :]  
  
# 调整图像大小  
def resize(image):  
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)  
  
# 转换RGB为YUV格式  
def rgb2yuv(image):  
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)  
  
# 图像预处理  
def preprocess(image):  
    image = crop(image)  
    image = resize(image)  
    image = rgb2yuv(image)  
    return image  
 
@sio.on('telemetry')  
def telemetry(sid, data):  
    if data:  
        # 汽车的当前转向角  
        #steering_angle = float(data["steering_angle"])  
        # 汽车的油门  
        #throttle = float(data["throttle"])  
        # 当前的速度  
        speed = float(data["speed"])  
        # 中心摄像头  
        image = Image.open(BytesIO(base64.b64decode(data["image"])))  
        try:  
            image = np.asarray(image)  # from PIL image to numpy array  
            image = preprocess(image)  # apply the preprocessing  
            image = np.array([image])  # the model expects 4D array  
  
            # 预测图像的转向  
            steering_angle = float(model.predict(image, batch_size=1))  
  
            # 根据速度调整油门，如果大于最大速度就减速，如果小于最低速度就加加速  
            if speed > MAX_SPEED:  
                speed_limit = MIN_SPEED  # slow down  
            else:  
                speed_limit = MAX_SPEED  
            throttle = 1.0 - steering_angle ** 2 - (speed / speed_limit) ** 2  
  
            print('{} {} {}'.format(steering_angle, throttle, speed))  
            send_control(steering_angle, throttle)  
        except Exception as e:  
            print(e)  
  
        # save frame  
        if image_folder != '':  
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]  
            image_filename = os.path.join(image_folder, timestamp)  
            array_to_img(image[0]).save('{}.jpg'.format(image_filename))  
    else:  
        sio.emit('manual', data={}, skip_sid=True)  
 
@sio.on('connect')  
def connect(sid, environ):  
    print("connect ", sid)  
    send_control(0, 0)  
  
def send_control(steering_angle, throttle):  
    sio.emit(  
        "steer",  
        data={  
            'steering_angle': steering_angle.__str__(),  
            'throttle': throttle.__str__()  
        },  
        skip_sid=True)  
  
if __name__ == '__main__':  
  
    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3  
    MAX_SPEED, MIN_SPEED = 25, 10  
    # 载入模型  
    model = load_model('model-001.h5')  
    image_folder = ''  
    # 设定图片缓存目录  
    if image_folder != '':  
        if os.path.exists(image_folder):  
            shutil.rmtree(image_folder)  
        os.makedirs(image_folder)  
  
    app = Flask(__name__)  
    app = socketio.Middleware(sio, app)  
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)  