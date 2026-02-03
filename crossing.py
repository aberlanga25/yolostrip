import tqdm
import glob
import logging
import configparser
import argparse
import cv2
import multiprocessing
import socket
import json
import datetime
import numpy as np
import supervision as sv
from queue import Queue
from YoloStrip.yolostrip import YoloStrip

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

UDP_QUEUE = Queue()
FRAME_QUEUE = Queue()

class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: dict[int, int] = {}
        self.counts: dict[int, dict[int, set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: list[sv.Detections],
        detections_out_zones: list[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]

class UDPClient(multiprocessing.Process):
    def __init__(self,
                 queue: Queue,
                 server_ip,
                 server_port):
        super().__init__()
        self.queue = queue
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.logger = logging.getLogger('UDPClient')

    def run(self):
        while True:
            message = self.queue.get()
            self.logger.debug('Got message')
            try:
                self.socket.sendto(message.encode(), (self.server_ip, self.server_port))
            except Exception as e:
                self.logger.error(f"Failed to send UDP message: {e}")


class ImageWriter(multiprocessing.Process):
    def __init__(self,
                 queue,
                 folder):
        super().__init__()
        self.queue = queue
        self.folder = folder
        self.logger = logging.getLogger('ImageWriter')

    def run(self):
        date = datetime.datetime.now(datetime.timezone.utc)
        video_path = f'{self.folder}/{date.strftime("%d-%m-%Y-%H")}.mp4'
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 1280))
        while True:
            if video_path != f'{self.folder}/{date.strftime("%d-%m-%Y-%H")}.mp4':
                try:
                    writer.release()

                    date = datetime.datetime.now(datetime.timezone.utc)
                    video_path = f'{self.folder}/{date.strftime("%d-%m-%Y-%H")}.mp4'
                    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 1280))
                except Exception as e:
                    self.logger.error(f"Failed to release video writer: {e}")
            image = self.queue.get()
            self.logger.debug('Got image')
            try:
                writer.write(image)
            except Exception as e:
                self.logger.error(f"Failed to write frame: {e}")


def detect(zones_file, model_file, stream_link, classes):
    with open(zones_file, 'r') as file:
        zones = json.load(file)

    ZONES_IN = []
    ZONES_OUT = []
    LAST_COUNT = {}
    i=0
    for zone in zones:
        x = np.array([[zone["firstX"],zone["firstY"]], [zone["secondX"],zone["secondY"]], [zone["thirdX"],zone["thirdY"]], [zone["fourthX"],zone["fourthY"]]])
        if zone['in'] == '1':
            ZONES_IN.append(sv.PolygonZone(x.astype(int),[sv.Position.BOTTOM_RIGHT]))
        else:
            ZONES_OUT.append(sv.PolygonZone(x.astype(int), [sv.Position.BOTTOM_RIGHT]))
            LAST_COUNT[i] = 0
            i+=0


    model = YoloStrip(model_file, imgsz=1280, classes=classes)

    tracker = sv.ByteTrack()
    detections_manager = DetectionsManager()

    cap = cv2.VideoCapture(stream_link)
   
    box_annotator = sv.BoxAnnotator(sv.Color.ROBOFLOW, thickness=2)
    label_annotator = sv.LabelAnnotator(sv.Color.ROBOFLOW)

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        pred = model(frame)
        detections = sv.Detections.from_yolov5(pred)
        detections = tracker.update_with_detections(detections)
        labels = [str(conf) for conf in detections.tracker_id]
        annotated_frame = frame.copy()

        detections_in_zones = []
        detections_out_zones = []

        for i, zone_in in enumerate(ZONES_IN):
            detections_in_zone = detections[zone_in.trigger(detections)]
            detections_in_zones.append(detections_in_zone)
            annotated_frame = sv.draw_polygon(annotated_frame, zone_in.polygon, COLORS.colors[i])

        for x, zone_out in enumerate(ZONES_OUT,start=i+1):
            detections_out_zone = detections[zone_out.trigger(detections)]
            detections_out_zones.append(detections_out_zone)
            annotated_frame = sv.draw_polygon(annotated_frame, zone_out.polygon, COLORS.colors[x])

        detections_manager.update(detections,detections_in_zones,detections_out_zones)
        date = datetime.datetime.now(datetime.timezone.utc)
        for zone_out_id, zone_out in enumerate(ZONES_OUT):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in detections_manager.counts:
                counts = detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(detections_manager.counts[zone_out_id][zone_in_id])
                    if LAST_COUNT[zone_out_id] != count:
                        UDP_QUEUE.put(f'{zone_out_id},{date.strftime("%d/%m/%Y, %H:%M:%S")}\n')
                        LAST_COUNT[zone_out_id] = count
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        annotated_frame = box_annotator.annotate(annotated_frame, detections)
        annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
        FRAME_QUEUE.put(annotated_frame)




if __name__ == '__main__':

    arg = argparse.ArgumentParser()
    arg.add_argument('--config','-c',default='./config.ini')

    pars = arg.parse_args()

    config = configparser.ConfigParser()
    config.read(pars.config)

    udp_server = config['MAIN']['SERVER']
    udp_port = int(config['MAIN']['PORT'])
    results_folder = config['MAIN']['OUTPUT']
    zones_file = config["MAIN"]['ZONES_FILE']
    model = config["MAIN"]['MODEL']
    stream = config["MAIN"]['STREAM']
    classes = [int(class_item) for class_item in config["MAIN"]["CLASSES"].split(',')] if 'CLASSES' in config['MAIN'] and config['MAIN']['CLASSES'].strip() else None

    udp_client = UDPClient(UDP_QUEUE, udp_server, udp_port)
    frame_writer = ImageWriter(FRAME_QUEUE, results_folder)
    udp_client.start()
    frame_writer.start()
    detect(zones_file, model, stream, classes)
    