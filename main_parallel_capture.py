import cv2
import threading
import numpy as np
import os
from ultralytics import YOLO
#import ollama
import base64
import asyncio
import threading
import queue
import time
import argparse
import os
import sys
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from datetime import datetime

########################################################
###  Class for handling real time streaming usin REST methods
########################################################
app = Flask(__name__)
CORS(app)  # allow all origins
# Number of cameras
NUM_CAMERAS = 8

# Camera indexes (adjust according to your system)
CAMERA_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
ENABLED_CAMERAS = [False,False,False,False,False,False,False,False]

# Dictionary to store the frames
frames = {i: None for i in CAMERA_IDS}
output_frames = [None, None]

# Lock for thread safety
lock = threading.Lock()

RESIZE_CAM_WIDTH = 640
RESIZE_CAM_HEIGHT = 480


# Queue for frames to be saved
frame_queue = queue.Queue()
stop_saver = False
threads = []
last_response = ""
chunks = []
best_frame = 0
### Define COCO keypoint pairs for skeleton connections
SKELETON_EDGES = [
    (5, 7), (7, 9),      # Left arm: shoulder → elbow → wrist
    (6, 8), (8, 10),     # Right arm: shoulder → elbow → wrist
    (5, 6),              # Shoulders connection
    (11, 12),            # Hips connection
    (5, 11), (6, 12),    # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

SKELETON_COLOR_EDGES = [
    (5, 6),              # Shoulders connection
    (11, 12),            # Hips connection
    (5, 11),     # Torso
    (11, 13),  # Left leg
    (12, 14)  # Right leg
]
CONF_THRESHOLD = 0.5  # confidence threshold (0-1)

os.makedirs("./temp/", exist_ok=True)

# Open a log file once for the whole program
log_file = open("./temp/frames_log.txt", "w")
log_lock = threading.Lock()

###################################################################3
def log_message(message):
    """
    Logs a message to console and to a text file (thread-safe).
    """
    with log_lock:
        print(message)
        log_file.write(message + "\n")
        log_file.flush()


##############################################################################
class TrackedPerson:
    def __init__(self, keypoints, confidences, track_id):
        self.keypoints = keypoints
        self.confidences = confidences
        self.colors = []
        self.colors_coords = []
        self.bbox = [0,0,0,0]
        self.track_id = track_id
        self.center = [0,0,0]
    def get_mean_conf(self,body_part="ALL"):
        
        if body_part == "HANDS":
            return (self.confidences[9] + self.confidences[10]) / 2
        else:
            cf = 0
            for idx, c in enumerate(self.confidences):
                cf += c / len(self.confidences)
            return cf 

        
    def compute_bbox(self, frame):
        """
        Compute bounding box (x, y, w, h) from a list of points.
        points: list of (x, y) tuples or numpy array shape (N, 2)
        """
        cn = [0, 0]
        counter = 0
        conf = 0
        points = []
        iw, ih = frame.shape[1], frame.shape[0]
        for idx, kp in enumerate(self.keypoints):
            if self.confidences[idx] > CONF_THRESHOLD:
                points.append((self.keypoints[idx][0]/iw, self.keypoints[idx][1]/ih))

        pts = np.array(points)
        x_min = np.min(pts[:, 0])
        y_min = np.min(pts[:, 1])
        x_max = np.max(pts[:, 0])
        y_max = np.max(pts[:, 1])
        self.bbox = [(x_min), (y_min), (x_max - x_min), (y_max - y_min)]
        return self.bbox
        
    def get_center(self, frame):
        cn = [0, 0]
        counter = 0
        conf = 0
        iw, ih = frame.shape[1], frame.shape[0]
        for idx, kp in enumerate(self.keypoints):
            if self.confidences[idx] > CONF_THRESHOLD:
                cn[0] += self.keypoints[idx][0]/iw
                cn[1] += self.keypoints[idx][1]/ih
                conf += self.confidences[idx]
                counter += 1
        
        if counter == 0:
            self.center = (0,0,0)
        else:
            self.center = (cn[0] / counter, cn[1] / counter, conf / counter)
            
        return self.center

    ###calculate like a finger print
    def compute_pose_colors(self, frame, hsv_frame):
        isk = 0
        for start, end in SKELETON_COLOR_EDGES:
            if self.confidences[start] > CONF_THRESHOLD and self.confidences[end] > CONF_THRESHOLD:
                pt1, pt2 = self.keypoints[start], self.keypoints[end]
                cn = ((pt1[0]+pt2[0])/2 , (pt1[1]+pt2[1])/2)
                hue, sat,val = get_average_hue_saturation(hsv_frame, int(cn[0]), int(cn[1]), 5)
                self.colors.append([float(hue), float(sat), float(val),self.confidences[start]])                
                self.colors_coords.append([int(cn[0]), int(cn[1])])
            else:
                self.colors.append([float(0), float(0), float(0),self.confidences[start]])                
                self.colors_coords.append([int(0), int(0)])



    def draw(self, image):
        """
        Draw skeleton on image given keypoints and confidences.
        keypoints: np.ndarray of shape (17, 2)
        confidences: np.ndarray of shape (17,)
        """
        # Draw edges only if both points have confidence above threshold
        neckX , neckY = self.keypoints[5]
        for start, end in SKELETON_EDGES:
            if self.confidences[start] > CONF_THRESHOLD and self.confidences[end] > CONF_THRESHOLD:
                pt1, pt2 = self.keypoints[start], self.keypoints[end]
                cv2.line(image, tuple(pt1.astype(int)), tuple(pt2.astype(int)), (0, 255, 0), 2)

        # Draw keypoints as circles only if confidence above threshold
        cv2.putText(image, f"id:{self.track_id}", (int(neckX),int(neckY-10) ), 1, 0.75, (255, 255, 255))

        for (x, y), conf in zip(self.keypoints, self.confidences):
            if conf > CONF_THRESHOLD:
                cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

## ---------------------------------------------------------------------------------------------
def send_image_to_VLLM( image , index):
    """
    Sends an asynchronous POST request to the API with the frame number.
    """
    try:
        image_path = f"./temp/test_sample{index}.jpg" 
        cv2.imwrite(image_path, image)

        # Read the image file and encode it in base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        now = time.time()
        # Define the message with the image
        messages = [
            {
                'role': 'user',
                'content': 'There should be a person in the scene. Answer which is the more similar activity that you can recognize from this list.' \
                           "0.Moving stuff or"    \
                            "1. Cleanning with duster or" \
                            "2. Talking by phone or" \
                            "3. Cleaning surfaces with cloth or" \
                            "4. Replace trash bags or" \
                            "5. Replace bed sheets or" \
                            "6. Mopping . " \
                            " Also, show me the confidence for each action. "
                            " If you can not identify the person answer with 'NONE' ",
                'images': [encoded_image]  # Pass the base64 encoded image
            }
        ]

        # Make the chat request to Ollama
        #response = ollama.chat(model='gemma3:4b', messages=messages)
        response = []
        # Print the model's response
        log_message("-------" + str(index) + "---------------" + response['message']['content'])
        
        last_response = response['message']['content']
        print(f"Ollama processing time: {time.time() - now} ")
       # cv2.imshow("ollama", image)
    except Exception as e:
        print(f"Exception: {e}")

## ----------------------------------------------------------------------------
def frame_saver_worker():
    """
    Background thread: pulls frames from queue and saves them to disk.
    """
    img_index = 0
    while not stop_saver or not frame_queue.empty():
        try:
            frame_number, frame = frame_queue.get(timeout=0.1)
           
            send_image_to_VLLM(frame,img_index)

            img_index += 1
           
        except queue.Empty:
            continue

## ----------------------------------------------------------------------------
def crop_from_bbox(image, bbox, offset):
    """
    Crop a region from an image given a bounding box.
    image: numpy array (BGR image from OpenCV)
    bbox: (x, y, w, h)
    Returns: cropped image (numpy array)
    """
    window = 0
    iw, ih = image.shape[1], image.shape[0]
    x, y, w, h = bbox
    x = (x-offset[0])*2 
    y = (y-offset[1])*2 
    w = 2*w
    h = 2*h
    return (int(x*iw),int(y*ih),int(w*iw),int(h*ih)) #image[int(y*ih-window):int((y+h)*ih+window), int(x*iw-window):int((x+w)*iw+window)]

detections_history = [{}, {}, {}, {}]
origins = [[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]]
##############################################################33
def compute_best_crop(persons,frames,img_small, frame_index):
    best_frame = 0
    best_conf = 0
    best_p = None
    best_origin = [0, 0]
    window = 50
    ## check weather the person is located
    for p in persons:
        cn = p.get_center(img_small)
        confidence =  p.get_mean_conf("HANDS")
        frame = 0
        if cn[0] < 0.5 and cn[1] < 0.5:
            frame = 0                    
        elif cn[0] < 0.5 and cn[1] >= 0.5:
            frame = 2     
        elif cn[0] > 0.5 and cn[1] < 0.5:
            frame = 1
        else:
            frame = 3
        detections_history[frame][frame_index] =  confidence
        if confidence > best_conf:
            best_conf = confidence
            best_p = p
            best_origin = origins[frame]
            best_frame = frame
    # Add frame to queue for background saving
    crop = None
    crop_img = None
    
    if best_conf > 0:
        #frame_queue.put((findex, frames[best_frame].copy()))
        crop = crop_from_bbox(frames[best_frame], best_p.bbox, best_origin)
        x,y,w,h = crop
      #  cv2.rectangle(frames[best_frame], (x, y), (x+w, y+h), (255, 0, 0), 4)
      #  cv2.imshow("best_Frame", crop)
        crop_img= frames[best_frame][max(0,y-window):y+h+window, max(0,x-window):x+w+window]
    
    return crop, best_frame, crop_img, best_conf


#####################################
## capture
def capture_camera(src, camera_id , findex):

    blank = False
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Camera {camera_id} not available. Using blank camera")
        ENABLED_CAMERAS[camera_id] = False
        blank = True
        cap = None
        
    else:
        print(f"Camera {camera_id}  available")
   # cap.set(cv2.CAP_PROP_POS_FRAMES, findex)
        ENABLED_CAMERAS[camera_id] = True
    ###      
    while True:

        if blank:
            frame = np.zeros((RESIZE_CAM_HEIGHT, RESIZE_CAM_WIDTH, 3), np.uint8)
            milliseconds_to_sleep = 25
            time.sleep(milliseconds_to_sleep / 1000)
        else:
            ret, frame = cap.read()
            if not ret:
                break
        frames[camera_id] = frame
        milliseconds_to_sleep = 10
        time.sleep(milliseconds_to_sleep / 1000)

       # cv2.waitKey(1)
    if cap is not None:
        cap.release()

#####################################################33
# Start a thread for each camera
def detect_persons(model, tracker, img_small, findex):
            # Run pose detection + tracking
    results = model.track(img_small, persist=False,verbose= False, tracker=tracker, conf=0.5)

    # Visualize
    annotated_frame = results[0].plot()

    # Show frame
    # cv2.imshow("Pose Tracking", annotated_frame)
        
    persons = []
    boxes = []

    for result in results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()  # shape: (num_people, 17, 2)
            if result.keypoints.conf is None:
                continue
            confidences = result.keypoints.conf.cpu().numpy()

            for person_kp, person_conf in zip(keypoints, confidences):
                if person_conf.mean() < 0.6:
                    continue
                tk_person = TrackedPerson(person_kp, person_conf, len(persons))
                tk_person.compute_bbox(img_small)
                persons.append(tk_person)
                boxes.append((tk_person.bbox[0], tk_person.bbox[1], tk_person.bbox[2],
                                tk_person.bbox[3], person_conf.mean()) )
    
    dets = []
    for box in boxes:
        x1, y1, w, h, score = box
        if score > 0.4:  # filter weak detections
            dets.append([x1, y1, x1+w, y1+h, score])
    
    crop = None
    

    crop, best_frame, crop_img, best_conf = compute_best_crop(persons, frames, img_small, findex)

    return persons, crop_img , best_frame, best_conf
        
####################################################333
def parallel_combine_videos(video_files,freq_ollama):
    global stop_saver, threads
    
    if not os.path.exists("yolo11m-pose.pt"):
        model = YOLO("yolo11m-pose.pt")
        model.export(format="onnx")

     # Load YOLO pose model (use 'yolov8s-pose.pt' for better accuracy)
    model = YOLO("yolo11m-pose.pt")
    tracker = "botsort.yaml"

   
    findex = 1000

     # Start background thread
    saver_thread = threading.Thread(target=frame_saver_worker, daemon=True)
    saver_thread.start()
   ##############################################################
    
    chunks =[ [0, 1, 2, 3] ,  [4, 5, 6, 7] ]
    
    for cam_id,cam_source in enumerate(video_files):
        t = threading.Thread(target=capture_camera, args=(cam_source, cam_id,findex))
        t.daemon = True
        t.start()
        threads.append(t)

    # Main loop: combine frames

    accumulated_detections = 0
    while True:

        for idCh, chunk in enumerate(chunks):

            best_conf = 0
            best_crop = None
            best_frame = None

            # Make sure we have all frames of this chunk
            if all(frames[cam_id] is not None for cam_id in chunk):
                # Resize frames for display (e.g., 320x240 each)
                resized = [cv2.resize(frames[cam_id], (RESIZE_CAM_WIDTH, RESIZE_CAM_HEIGHT)) for cam_id in chunk]

                # Combine into 2x2 grid
                top = cv2.hconcat([resized[0], resized[1]])
                bottom = cv2.hconcat([resized[2], resized[3]])
                combined = cv2.vconcat([top, bottom])

                persons, crop, sel_frame, conf = detect_persons(model, tracker, combined, findex)
                                    
                if conf > best_conf:
                    best_conf = conf
                    best_crop = crop
                    best_frame =resized[ sel_frame]

                for pk in persons:
                    pk.draw(combined)

                
                # Get the current datetime object
                current_datetime = datetime.now()
                format_string = "%Y-%m-%d %H:%M:%S"
                time_string = current_datetime.strftime(format_string)
                cv2.putText(combined, time_string, (int(20), int(50)), 1,1, (255, 255, 255))
                
                
                cv2.putText(combined, f"id:{findex}", (int(20),int(20) ), 1, 0.75, (255, 255, 255))

                
        

                cv2.imshow(f"YOLO + SORT Tracking {idCh}", combined)
            # print (f"update {findex}")

                if accumulated_detections > freq_ollama and crop is not None:
                    frame_queue.put((findex, frames[best_frame].copy()))
                    accumulated_detections = 0
            
            
                # Draw keypoints and skeleton
                #annotated_frame = results[0].plot()
            if best_crop is not None and (best_crop.shape[0] * best_crop.shape[1]) > 0:
                cv2.imshow(f"crop ", best_crop)
                accumulated_detections += 1
            output_frames[0] = best_frame
            k = cv2.waitKey(20) 
            if k & 0xFF == ord("q"):
                break
            if k & 0xFF == ord("1"):
                selected_view = 0
            if k & 0xFF == ord("2"):
                selected_view = 1
            if k & 0xFF == ord("3"):
                selected_view = 2
            if k & 0xFF == ord("4"):
                selected_view = 3
                
            findex += 1

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

@app.route("/vllm")
def get_vllm_response():
  
  
    return jsonify({ "vllm": last_response})
    

def generate_frames(idSource=1):
   
    while True:
        
        ################################
        frame = output_frames[0]
        if frame is None:
            print (f"Waiting to have a frame at {idSource}")
            milliseconds_to_sleep = 1000
            time.sleep(milliseconds_to_sleep / 1000)
            continue
        resized = cv2.resize(frame, (RESIZE_CAM_WIDTH, RESIZE_CAM_HEIGHT))
        
        current_datetime = datetime.now()
        format_string = "%Y-%m-%d %H:%M:%S"
        time_string = current_datetime.strftime(format_string)
        cv2.putText(resized, time_string, (int(20), int(50)), 1,1, (255, 255, 255))

       
        #frame = process_frame(frame, idx)
        ret, buffer = cv2.imencode('.jpg', resized)
        frame_bytes = buffer.tobytes()
        milliseconds_to_sleep = 10
        time.sleep(milliseconds_to_sleep / 1000)
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/status")
def get_status():
    
    return jsonify({"status": 200})


@app.route("/events")
def get_events():
    events = []
    return jsonify({ "events": events})
    


@app.route("/tags/<int:idx>")
def get_tags(idx):
    tags = []    
    return jsonify({"feed": idx, "tags": tags})

@app.route("/video_feed<int:idx>")
def video_feed(idx):
   return Response(generate_frames(0),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
   


######################################################################
def parse_args():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description="Parse command-line arguments for video processing.")
   # Define command-line arguments with optional flags
    parser.add_argument("--input_videos", default = "E:/Resources/Novathena/multicam_2/", help="Path to the input file")
    parser.add_argument("--freq_ollama", default =100, help="Freq to call Ollama")
    parser.add_argument("--port", default = "5101", help="Port for exposing")
    # Parse the arguments
    args = parser.parse_args()

    # Return the parsed arguments
    return args

if __name__ == "__main__":
    args = parse_args()

    if sys.platform == 'win32':
        print("Running on Windows")
        path = "E:/Resources/Novathena/multicam_2/"
        video_files = [path + "cam1_trimmed.mp4", path + "cam2_trimmed.mp4",
                        path + "cam3_trimmed.mp4", path + "cam4_trimmed.mp4", path + "cam6.mp4", path + "cam7.mp4",
                        ## fill with empty
                        "blank", "blank"]
    else:
        video_files =  [ "rtsp://root:Nova2022**@10.10.1.107:554/axis-media/media.amp?streamprofile=stream1",
                        "rtsp://root:Nova2022**@10.10.1.103:554/axis-media/media.amp?streamprofile=stream1",
                    "rtsp://root:Nova2022**@10.10.1.106:554/axis-media/media.amp?streamprofile=stream1",
                    "rtsp://root:Nova2022**@10.10.1.105:554/axis-media/media.amp?streamprofile=stream1"]
  
    print ("-------------------------------------------------------")
    print("Version of parallel combined Cameras . 30set2025 version chunks")
    print("-------------------------------------------------------")
    
    ### start parallel processors
    #for cam_id,cam_source in enumerate(sources):
    t = threading.Thread(target=parallel_combine_videos, args=(video_files, int(args.freq_ollama), ))
    t.daemon = True
    t.start()
    threads.append(t)

    app.run(host="0.0.0.0", port=args.port, debug=False)
