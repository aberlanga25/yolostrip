import tqdm
import glob
import cv2
import os
import json
import datetime
import numpy as np
import supervision as sv
from YoloStrip.yolostrip import YoloStrip

FOLDER = '/media/aberlanga/Elements/Pedestrians Zaragoza/'

DAYS = sorted(os.walk(FOLDER + 'data/'))[1:]
COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])

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

with open('./students.json', 'r') as file:
    zones = json.load(file)

ZONES_IN = []
ZONES_OUT = []

for zone in zones:
    x = np.array([[zone["firstX"],zone["firstY"]], [zone["secondX"],zone["secondY"]], [zone["thirdX"],zone["thirdY"]], [zone["fourthX"],zone["fourthY"]]])
    if zone['in'] == '1':
        ZONES_IN.append(sv.PolygonZone(x.astype(int),[sv.Position.BOTTOM_RIGHT]))
    else:
        ZONES_OUT.append(sv.PolygonZone(x.astype(int), [sv.Position.BOTTOM_RIGHT]))

model = YoloStrip('../yolov5/zoo/CrowdHuman-x6.pt', imgsz=1280, classes=[0])

for (day, _, _) in DAYS:
    tracker = sv.ByteTrack()
    detections_manager = DetectionsManager()
    d = day.split('/')[-1]
    print(d)
    os.makedirs(FOLDER + f'results/{d}', exist_ok=True)
    with open(FOLDER + f'results/{d}.csv', 'w+') as csv:
        for video in tqdm.tqdm(sorted(glob.glob(day+'/*.mp4')),position=0):
            video_info = sv.VideoInfo.from_video_path(video)
            cap = cv2.VideoCapture(video)
            filepath = os.path.basename(video)
            date = filepath.split('_')[:3]
            date = datetime.datetime.strptime(date[1]+date[2],'%Y%m%d%H%M%S')
            box_annotator = sv.BoxAnnotator(sv.Color.ROBOFLOW, thickness=2)
            label_annotator = sv.LabelAnnotator(sv.Color.ROBOFLOW)
            with sv.VideoSink(FOLDER + f'results/{d}/{filepath}.mp4', video_info) as sink:
                for num_frame in tqdm.tqdm(range(video_info.total_frames),position=1):

                    ret, frame = cap.read()
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

                    for zone_out_id, zone_out in enumerate(ZONES_OUT):
                        zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
                        if zone_out_id in detections_manager.counts:
                            counts = detections_manager.counts[zone_out_id]
                            for i, zone_in_id in enumerate(counts):
                                count = len(detections_manager.counts[zone_out_id][zone_in_id])
                                csv.write(f'{num_frame},{zone_out_id},{count},{date.strftime("%d/%m/%Y, %H:%M:%S")}\n')
                                text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                                annotated_frame = sv.draw_text(
                                    scene=annotated_frame,
                                    text=str(count),
                                    text_anchor=text_anchor,
                                    background_color=COLORS.colors[zone_in_id],
                                )

                    annotated_frame = box_annotator.annotate(annotated_frame, detections)
                    annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)


                    sink.write_frame(annotated_frame)

