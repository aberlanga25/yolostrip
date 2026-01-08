import tqdm
import glob
import cv2
import os
import numpy as np
import supervision as sv
from YoloStrip.yolostrip import YoloStrip

DAYS = './dataset/StudentsZaragoza/*'

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

model = YoloStrip('../yolov5/zoo/CrowdHuman-x6.pt',imgsz=1280)
tracker = sv.ByteTrack()

for video in glob.glob(VIDEOS):
    video_info = sv.VideoInfo.from_video_path(video)
    cap = cv2.VideoCapture(video)
    filepath = os.path.basename(video)
    box_annotator = sv.BoxAnnotator(sv.Color.ROBOFLOW, thickness=2)
    label_annotator = sv.LabelAnnotator(sv.Color.ROBOFLOW)
    with sv.VideoSink(f'runs/Students/{filepath}.mp4', video_info) as sink:
        for num_frame in tqdm.tqdm(range(video_info.total_frames)):

            ret, frame = cap.read()
            pred = model(frame)
            detections = sv.Detections.from_yolov5(pred)
            detections = tracker.update_with_detections(detections)
            labels = [str(conf) for conf in detections.tracker_id]
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)

            sink.write_frame(annotated_frame)

