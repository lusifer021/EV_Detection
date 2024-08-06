import numpy as np
import torch
import os
import sys
import argparse
sys.path.append(os.path.abspath('../yolov5'))
from utils.general import non_max_suppression, scale_coords
# from ai_core.object_detection.yolov5_custom.od.data.datasets import letterbox
from typing import List
# from dynaconf import settings
from models.experimental import attempt_load
import cv2
import pandas as pd
import tqdm
import time
import pickle

class Detection:
    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
        cwd = os.path.dirname(__file__)
        self.device=device
        self.char_model, self.names = self.load_model(weights_path)
        self.size=size
        
        self.iou_thres=iou_thres
        self.conf_thres=conf_thres

    def detect(self, frame):
        
        results, resized_img = self.char_detection_yolo(frame)

        return results, resized_img
    
    def preprocess_image(self, original_image):

        resized_img = self.ResizeImg(original_image,size=self.size)
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img
    
    def char_detection_yolo(self, image, classes=None, \
                            agnostic_nms=True, max_det=1000):

        img,resized_img = self.preprocess_image(image.copy())
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(pred, conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=classes,
                                            agnostic=agnostic_nms,
                                            multi_label=True,
                                            labels=(),
                                            max_det=max_det)
        results=[]
        for i, det in enumerate(detections):
            # det[:, :4]=scale_coords(resized_img.shape,det[:, :4],image.shape).round()
            det=det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    # xc,yc,w_,h_=(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])
                    result=[self.names[int(cls)], str(conf), (xyxy[0],xyxy[1],xyxy[2],xyxy[3])]
                    results.append(result)
        # print(results)
        return results, resized_img
        
    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        # print(h1, w1, _)
        h, w = size
        if w1 < h1 * (w / h):
            # print(w1/h1)
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
    def load_model(self,path, train = False):
        # print(self.device)
        model = attempt_load(path, map_location=self.device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if train:
            model.train()
        else:
            model.eval()
        # print(names)
        return model, names
    def xyxytoxywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # x center
        y[1] = (x[1] + x[3]) / 2  # y center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y
    


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='Vietnamese_imgs', help='file/dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt



def convert_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

green_lower = np.array([35, 50, 50])
green_upper = np.array([85, 255, 255])
white_lower = np.array([0, 0, 200])
white_upper = np.array([180, 30, 255])
yellow_lower = np.array([20, 50, 50])
yellow_upper = np.array([30, 255, 255])

def create_color_masks(hsv_image):
    mask_green = cv2.inRange(hsv_image, green_lower, green_upper)
    mask_white = cv2.inRange(hsv_image, white_lower, white_upper)
    mask_yellow = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    return mask_green, mask_white, mask_yellow

def detect_dominant_color(mask_green, mask_white, mask_yellow):
    green_count = cv2.countNonZero(mask_green)
    white_count = cv2.countNonZero(mask_white)
    yellow_count = cv2.countNonZero(mask_yellow)

    if green_count > white_count and green_count > yellow_count and green_count > 50:
        return "green"
    elif white_count > yellow_count:
        return "white"
    else:
        return "yellow"



if __name__ == '__main__':
    opt = parse_opt()
    
    char_model=Detection(size=opt.imgsz,weights_path=opt.weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    root_dir=r"C:\Users\Dell\Downloads\Downloads\Closeup"
    if os.path.exists(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\results.csv"):
        results_df = pd.read_csv(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\results.csv")
    else:
        results_df = pd.DataFrame(columns=['video_file_name', 'vehicle_class', 'timestamp'])

    if os.path.exists(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\videos_done_list"):
        with open(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\videos_done_list", "rb") as fp:   # Unpickling
            videos_done = pickle.load(fp)
    else:
        videos_done = []
        with open(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\videos_done_list", "wb") as fp:   #Pickling
            pickle.dump(videos_done, fp)

        with open(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\videos_done_list", "rb") as fp:   # Unpickling
            videos_done = pickle.load(fp)
        
        # print(videos_done)

    # print(videos_done)
    # sys.exit()
    video_files = []
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.mov') or file.endswith('.mp4'):  # Add other video formats if needed
                video_files.append(os.path.join(subdir, file))
    
    # print(video_files)
    for video_file_path in tqdm.tqdm(video_files, desc="Processing videos"):
            
        # if file.endswith('.mp4') or file.endswith('.avi'):  # Add other video formats if needed
        # video_file_path = os.path.join(subdir, file)
        # video_results_df = process_video(video_file_path)
        # all_results_df = pd.concat([all_results_df, video_results_df], ignore_index=True)

# img_names=os.listdir(path)
        new_video_name = video_file_path[:-4]+"_.mp4"
        if os.path.exists(new_video_name) == False:
            os.system(f"ffmpeg -i {video_file_path} -r 25 {new_video_name}")

        print(new_video_name)
        cap = cv2.VideoCapture(new_video_name)
        base_name_with_extension = os.path.basename(video_file_path)
        
        # video_files_completed = results_df.video_file_name.values.tolist()[:-1]

        if(base_name_with_extension in videos_done):
            continue
        else:
            videos_done.append(base_name_with_extension)

        print(video_file_path)
        if not cap.isOpened():
            print("Error: Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(total_frames)
        print(fps)
        frame_interval = int(fps // 3)

        # frame_interval = int(fps//3)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_count = 0
        # start_time = 21  # 1 minute
        # end_time = 25

        # start_frame = int(start_time * fps)
        # end_frame = int(end_time * fps)

        # cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        # frame_count = start_frame
        # for img_name in img_names:
        #     img=cv2.imread(os.path.join(path,img_name))
        #     results, resized_img=char_model.detect(img.copy())
        # print(file)
        # out_video_name = video_file_path.split("/")[-1]
        # output_video_path = f'{out_video_name}'
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec
        # out = cv2.VideoWriter(output_video_path, fourcc, fps, (1280, 1280))
        # resized_images = []
        # start_time = time.time()
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_count += 1
            if(frame_count % 100 == 0):
                print(frame_count)
            if not ret:
                print("End of video or error.")
                break
            if frame_count % frame_interval == 0:
            # if True:
                # print(1)
                results, resized_img=char_model.detect(frame.copy())


                # print(resized_img.shape)

                # print(results)
                for name,conf,box in results:
                    if name == "car":
                        resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                (255, 0, 255), 2)
                        resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
                        
                    if name == "rectangle license plate":
                        flag = 0
                        # print("hello")
                        # resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        #                         (255, 0, 255), 2)
                        # resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
                        # print("box",box)
                        for name_car,conf_car,box_car in results:
                            if name_car == "car":
                                # print("box_car",box_car)
                                #check if box is inside box_car
                                x1,y1,x2,y2=box
                                x1_car,y1_car,x2_car,y2_car=box_car
                                if x1>x1_car and y1>y1_car and x2<x2_car and y2<y2_car:
                                    flag = 1
                                    break
                        
                        if flag == 1:
                            # print("hello")
                            # resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                            #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            #                         (255, 0, 255), 2)
                            resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
                            # cut out the license plate
                            license_plate = resized_img[int(box[1]):int(box[3]),int(box[0]):int(box[2])]
                            hsv_image = convert_to_hsv(license_plate)
                            mask_green, mask_white, mask_yellow = create_color_masks(hsv_image)
                            dominant_color = detect_dominant_color(mask_green, mask_white, mask_yellow)
                            # print("Dominant color of license plate: ", dominant_color)
                            if(dominant_color == "green"):
                                timestamp = frame_count / fps
                                # print("Timestamp of green license plate detection:", timestamp)
                                # resized_img=cv2.putText(resized_img, "{}".format("green"), (int(box[0]), int(box[1])-3),
                                #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                #                 (255, 0, 255), 2)
                                new_row = {
                                'video_file_name': base_name_with_extension,
                                'vehicle_class': "car",
                                'timestamp': timestamp
                                }
                                results_df =  pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
                                # resized_img = cv2.putText(resized_img, "{}".format(dominant_color), (int(box[0]), int(box[1])-3),
                                #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                #                     (255, 0, 255), 2)
                                # cv2.imwrite(f'{out_video_name.split(".")[0]}_{frame_count}.jpg', marked_frame)
                                # out.write(marked_frame)
                                # cv2.imshow("marked_frame", marked_frame)
                            # else:
                                # out.write(resized_img)
                                # cv2.imshow("marked_frame", resized_img)
                        # else:
                            # out.write(resized_img)       
                            # if cv2.waitKey(1) & 0xFF == ord('q'):
                            #     break
                    # else:
                    # resized_images.append(resized_img)
                # timestamp = frame_count / fps
                # cv2.imwrite(f'./output_frames/{video_file_path.split("/")[-1].split(".")[0]}_{timestamp}.jpg', resized_img)
            # out.write(resized_img)
            # # frame_count += 1
            # next_frame_time = start_time + frame_count * frame_interval
            
            # # Sleep if necessary to maintain frame rate
            # current_time = time.time()
            # if current_time < next_frame_time:
            #     time.sleep(next_frame_time - current_time)
        # for i in resized_images:
        #     out.write(i)
                
                # frame_count += 1
        results_df.to_csv(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\results.csv", index=False)
        with open(r"C:\Users\Dell\Desktop\Character-Time-series-Matching\Vietnamese\videos_done_list", "wb") as fp:   #Pickling
            pickle.dump(videos_done, fp)
        # if os.path.exists(new_video_name):
        #     import shutil
        #     shutil.rmtree(new_video_name)
        #     print(f"File '{new_video_name}' deleted successfully.")
        # else:
        #     print(f"File '{new_video_name}' does not exist.")
        
            
        cap.release()
        # out.release()
        cv2.destroyAllWindows()


    # results_df.to_csv('results.csv', index=False)
    # cap.release()
    # # out.release()
    # cv2.destroyAllWindows()

        # if not os.path.exists(os.path.join('out')):
        #     os.makedirs(os.path.join('out'))
        # cv2.imwrite(os.path.join('out',img_name),resized_img)
