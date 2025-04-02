import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

cap = cv2.VideoCapture('12364947_1920_1080_30fps.mp4')

model = YOLO('yolov8m.pt')
tracker = Sort(max_age=60*3, min_hits=3, iou_threshold=0.5)

#video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

cross_line_1 = 0
cross_list1 = []
cross_line_2 = 0
cross_list2 = []

previous_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    line_1_y = 850
    line_2_y = 750

    cv2.line(frame, (220, line_1_y), (1080, line_1_y), (109, 34, 166), 2)
    cv2.line(frame, (1150, line_2_y), (1700, line_2_y), (34, 105, 255), 2)

    for result in results:
        if result.boxes.conf[0] < 0.5:
            continue

        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        tracks = tracker.update(boxes)

        for track in tracks:
            x1, y1, x2, y2, track_id = map(int, track)
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            if track_id in previous_positions:
                prev_y = previous_positions[track_id]

                if prev_y < line_1_y and center[1] >= line_1_y and center[0] > 220 and center[0] < 1080:
                    if track_id not in cross_list1:
                        cross_list1.append(track_id)

                if prev_y < line_2_y and center[1] >= line_2_y and center[0] > 1150 and center[0] < 1700:
                    if track_id not in cross_list2:
                        cross_list2.append(track_id)


            previous_positions[track_id] = center[1]

            cv2.circle(frame, center, 3, (0, 0, 255), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, str(track_id), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.putText(frame, f'Cross_1: {len(cross_list1)}', (220, 800), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,115,239), 2)
    cv2.putText(frame, f'Cross_2: {len(cross_list2)}', (1120, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (30,115,239), 2)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print('Process completed')
