import cv2
from deepface import DeepFace

def highlightFace(net, frame, conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
faceNet = cv2.dnn.readNet(faceModel, faceProto)


video = cv2.VideoCapture(0)
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    
    if not faceBoxes:
        print("Лица не распознаны")
    else:
        print("Лица распознаны")
        for faceBox in faceBoxes:
            x1, y1, x2, y2 = faceBox
            face_img = frame[y1:y2, x1:x2]
            result_list = DeepFace.analyze(img_path=face_img, actions=['race', 'gender', 'emotion'], enforce_detection=False)
            for result_dict in result_list:
                dominant_race = max(result_dict['race'], key=result_dict['race'].get)
                dominant_gender = max(result_dict['gender'], key=result_dict['gender'].get)
                dominant_emotion = max(result_dict['emotion'], key=result_dict['emotion'].get)
                cv2.putText(resultImg, f"Race: {dominant_race}", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(resultImg, f"Gender: {dominant_gender}", (x1, y1 - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.putText(resultImg, f"Emotion: {dominant_emotion}", (x1, y1 - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow("Face detection", resultImg)