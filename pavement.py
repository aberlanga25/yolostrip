import cv2
import glob
import os
import bisect
import math
import simplekml
import zipfile
import supervision as sv
from tqdm import tqdm
from datetime import datetime, timedelta, date
from YoloStrip.yolostrip import YoloStrip
from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor

def put_text(frame, text, num=1):
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    height, _, _ = frame.shape
    width = (text_width // 2) + 10
    text_anchor = sv.Point(x=width,y=10+(30*num))
    frame = sv.draw_text(frame, text, text_anchor, background_color=sv.Color(r=255, g=255, b=255))
    return frame


model = YoloStrip("./runs/train/RDD4/weights/best.pt", imgsz=1280, conf_thres=0.5, classes=[3])
videos = glob.glob('../YoloSupervision/Pavement/2025-11-11/*')

for video in videos:
    filepath = os.path.basename(video)
    print(filepath)
    extractor = GoProTelemetryExtractor(video)
    extractor.open_source()

    gps, gps_t = extractor.extract_data("GPS9")
    accl, accl_t = extractor.extract_data("ACCL")
    gyro, gyro_t = extractor.extract_data("GYRO")

    extractor.close_source()
    video_info = sv.VideoInfo.from_video_path(video)
    cap = cv2.VideoCapture(video)
    color_pallete = sv.ColorPalette([sv.Color.BLACK,sv.Color.ROBOFLOW])
    box_annotator = sv.BoxAnnotator(color_pallete, thickness=2)
    label_annotator = sv.LabelAnnotator(color_pallete)
    kml = simplekml.Kml()
    start_date = date(2000, 1, 1)

    every_accl = round(video_info.total_frames / len(accl_t), 0)
    every_gyro = round(video_info.total_frames / len(gyro), 0)

    os.makedirs(f'runs/{filepath}',exist_ok=True)
    with sv.VideoSink(f'runs/{filepath}/{filepath}.mp4', video_info) as sink:
        for num_frames in tqdm(range(video_info.total_frames)):
            x = int(num_frames / 3)
            accel_idx = bisect.bisect_left(accl_t, gps_t[x])
            gyro_idx = bisect.bisect_left(gyro_t, gps_t[x])
            #x_gyro = int(num_frames / every_gyro)

            lat, long, alt, speed, _, days, sec, _,_ = gps[x]
            time = gps_t[x]

            gyro_z, gyro_x, gyro_y = gyro[gyro_idx]
            accl_z, accl_x, accl_y = accl[accel_idx]

            current_date = start_date + timedelta(days=days)
            fractional_part, integer_part = math.modf(sec)
            str_time = f'{str(current_date)} {str(timedelta(seconds=integer_part))}{str(round(fractional_part,3))[1:]}'

            ret, frame = cap.read()
            results = model(frame)
            
            detections = sv.Detections.from_yolov5(results)
            labels = [f'{str(conf)},{str(model.names[name])}' for conf, name in zip(detections.confidence, detections.class_id)]
            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(annotated_frame, detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections, labels)
            str_gps = f'GPS: {round(lat,5)},{round(long,5)},{round(alt,5)}  SPD: {speed} Frame: {num_frames}'
            str_accl = f'ACCL: {round(accl_x,5)},{round(accl_y,5)},{round(accl_z, 5)}  GYRO: {round(gyro_x,5)},{round(gyro_y, 5)},{round(gyro_z, 5)}'
            #str_gyro = f''
            #str_speed = f''
            
            #annotated_frame = put_text(annotated_frame, str_gyro)
            annotated_frame = put_text(annotated_frame, str_accl, 1)
            #annotated_frame = put_text(annotated_frame, str_speed, 3)
            annotated_frame = put_text(annotated_frame, str_gps, 0)

            if len(detections) > 0:
                pnt = kml.newpoint(name=f"{num_frames}", coords=[(long, lat)])
                for x, dets in enumerate(detections):
                    if dets[3] == 3:
                        os.makedirs(f'runs/{filepath}/frames',exist_ok=True)
                        cv2.imwrite(f'runs/{filepath}/frames/{dets[3]}-{x}-{filepath}-{num_frames}.png', annotated_frame)
            
            sink.write_frame(annotated_frame)
            
    kml_content = kml.kml()

    kmz_filename = f"runs/{filepath}/map.kmz"
    kml_filename_in_kmz = "doc.kml" # Standard name for the main KML file in a KMZ

    with zipfile.ZipFile(kmz_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(kml_filename_in_kmz, kml_content)

