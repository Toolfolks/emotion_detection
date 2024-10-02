import cv2
from ultralytics import YOLO
import supervision as sv
import cvzone

model_face = YOLO("models/yolov8l-face.pt")
model_cls = YOLO("models/yolo11m-cls-affectnet.pt")
classes = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
tracker = sv.ByteTrack()

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    # Stream is online, so proceed with reading the frames
    for results in model_face(source=frame, verbose=False, stream=True, show=False, conf=0.1):
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        frame = results.orig_img
        
        # Iterar sobre as detecções
        for bbox, tracker_id in zip(detections.xyxy, detections.tracker_id):
            print(f'bbox: {bbox}')
            # Extrair coordenadas do bounding box
            x1, x2, x3, x4 = map(int, bbox)
            w, h = x3 - x1, x4 - x2

            if len(bbox):
                # Ajustar as coordenadas do recorte para ser 50% maior em cada dimensão
                expansion_factor = 1.5

                # Calcula as dimensões do retângulo bounding box original
                width, height = x3 - x1, x4 - x2

                # Calcula as novas dimensões do retângulo aumentado e as coordenadas do ponto inicial
                expanded_width, expanded_height = int(width * expansion_factor), int(height * expansion_factor)
                expanded_x1, expanded_x2 = max(0, x1 - (expanded_width - width) // 2), max(0, x2 - (expanded_height - height) // 2)
                expanded_x3, expanded_x4 = min(frame.shape[1], expanded_x1 + expanded_width), min(frame.shape[0], expanded_x2 + expanded_height)

                # Recorte com dimensões ajustadas
                face = frame[expanded_x2:expanded_x4, expanded_x1:expanded_x3]

                # predict emotion
                for result_cls in model_cls(source=face, verbose=False):
                    class_cls = classes[result_cls.probs.top1]
                    print(f'probs: {class_cls}')

                    # Desenhar o bounding box
                    cvzone.cornerRect(frame, (x1, x2, w, h), l=10, t=2)

                    # Adicionar texto com a classe
                    cvzone.putTextRect(frame,
                                f'#{class_cls}',  # Modifique 'results.names' conforme necessário
                                (max(0, x1), max(35, x2)),  # Ajustar a posição do texto conforme necessário
                                scale=0.5, thickness=1, colorR=(0, 255, 0),
                                colorT=(40, 40, 40),
                                font=cv2.FONT_HERSHEY_DUPLEX,
                                offset=5)
                    
    cv2.imshow('frame', frame)

    key = cv2.waitKey(1) & 0xFF

    # stop the frame
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()