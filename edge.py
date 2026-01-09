import argparse
import numpy as np
import base64
import requests
import datetime
import logging
from requests.auth import  HTTPDigestAuth
import json
from YoloStrip.yolostrip import YoloStrip
from io import BytesIO
import cv2
import supervision as sv
import configparser
import json
import time



MODEL = 'yolov5x6.pt'

arg = argparse.ArgumentParser()
arg.add_argument('--config','-c',default='./config.ini')

pars = arg.parse_args()

config = configparser.ConfigParser()
config.read(pars.config)

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

model = YoloStrip(MODEL, imgsz=1280, classes=[2,5,7])

names = model.names

box_annotator = sv.BoxAnnotator(color=COLORS)
label_annotator = sv.LabelAnnotator(
            color=COLORS, text_color=sv.Color.BLACK,text_padding=5,text_scale=0.4)

text_anchor_p = sv.Point(x=50,y=700)

for con in config.sections():
    try:
        POLYGON_FILE_PATH = config[con]['polygon']
        STREAM_FILE_PATH = config[con]['stream']
        PARKING_ID = config[con]['parking_id']
        URL = config[con]['server']
        MODEL_C = config[con]['model']
        SAVE_IMG = config[con]['save_img']
        CAMERA_ID = config[con]['camera_id']
        PTZ = config[con]['pt']

        if MODEL_C != MODEL:
            model = YoloStrip(MODEL_C)
            MODEL = MODEL_C

        if PTZ:
            print(PTZ)
            response = requests.get(PTZ)
            time.sleep(5)
            print(f"Slept 5 PTZ Camera {response.status_code}")

        with open(POLYGON_FILE_PATH, 'r') as file:
            zones = json.load(file)

        list_zones = []
        list_locs = []
        list_zone_ids = []

        for zone in zones:
            x = np.array([[zone["firstX"],zone["firstY"]], [zone["secondX"],zone["secondY"]], [zone["thirdX"],zone["thirdY"]], [zone["fourthX"],zone["fourthY"]]])
            list_zones.append(x.astype(float).astype(int))

            y = int(zone["loc"])
            list_locs.append(y)
            list_zone_ids.append(int(zone["id"]))

        zones_in = []
        for loc, polygon in zip(list_locs, list_zones):
            if loc==2:
                trig_anchor = (sv.Position.BOTTOM_LEFT,)
            elif loc==1:
                trig_anchor = (sv.Position.BOTTOM_CENTER,)
            else:
                trig_anchor = (sv.Position.BOTTOM_RIGHT,)

            zones_in.append(sv.PolygonZone(polygon=polygon, triggering_anchors=trig_anchor, frame_wh = (1280, 720)))
        if PTZ:
            response = requests.get(STREAM_FILE_PATH)
        else:

            response = requests.get(STREAM_FILE_PATH, auth=HTTPDigestAuth('user','passwrd'))
        print(response.status_code)
        img_stream = BytesIO(response.content)
        img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)

        results = model(img)
        detections = sv.Detections.from_yolov5(results)
        del results

        detections_in_zones = []

        labels = [str(model.names[x]) for x in detections.class_id]

        frame = img.copy()

        frame = box_annotator.annotate(frame, detections)
        frame = label_annotator.annotate(frame, detections, labels)
        x = 0
        parking_spots = []
        for i, zone_in in enumerate(zones_in):

            detections_in_zone = detections[zone_in.trigger(detections)]

            detections_in_zones.append(detections)
            x += 1 if len(detections_in_zone) > 0 else 0
            frame = sv.draw_polygon(frame, zone_in.polygon, COLORS.colors[0])
            polygon_center = sv.get_polygon_center(zone_in.polygon)
            text_anchor = sv.Point(x=polygon_center.x, y=polygon_center.y)
            if list_locs[i] == 0:
                str_loc = "BR"
            elif list_locs[i] == 1:
                str_loc = "BC"
            else:
                str_loc = "BL"
            frame = sv.draw_text(frame,str(list_zone_ids[i]) + " " + str_loc,text_anchor)
            park = {'parking_space_id':str(list_zone_ids[i]), 'status': 1 if len(detections_in_zone) > 0 else 0}
            parking_spots.append(park)


        frame = sv.draw_text(frame, f'{len(detections)} / {x}', text_anchor_p, background_color=sv.Color(r=255, g=255, b=255))

        #cv2.imwrite('./test.png', frame)

        #with open('./test.png', 'rb') as file_image:
        ret, buffer = cv2.imencode('.jpg', frame)
        imb64 = base64.b64encode(buffer.tobytes()).decode('utf-8')

        date = datetime.datetime.now(datetime.timezone.utc)

        data = {'parking_id':str(PARKING_ID),
                'parking_spaces': parking_spots,
                'image': imb64,
                'update_time': date.strftime("%Y-%m-%d %H:%M:%S"),
                'camera_id': str(CAMERA_ID)}

        j = json.dumps(data)

        try:
            response = requests.post(URL, data=j)
        except Exception as e:
            print(e)

        if SAVE_IMG:
            cv2.imwrite(SAVE_IMG + f'/{CAMERA_ID}/{date.strftime("%Y-%m-%d_%H-%M")}.png', frame)
    except  Exception as e:
        print(e)

