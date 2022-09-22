import cv2
import time
import os
import argparse

CONFIDENCE_THRESHOLD = 0.2
#NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)] 

modelfolder = os.path.join(os.getenv('HOME'),'models')
datafolder = os.path.join(os.getenv('HOME'),'bags')
class Dnn():
    def __init__(self,inp,theshold):

        self.NMS_THRESHOLD = theshold
          
        self.class_names = []
        with open(os.path.join(inp,"coco.names"), "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

       
        #vc = cv2.VideoCapture(args.datafolder)
        net = cv2.dnn.readNet(os.path.join(inp,"yolov4-mish-416.weights"), os.path.join(modelfolder,"yolov4-mish-416.cfg"))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)   

    def detect(self, frame):
        classes, scores, boxes = self.model.detect(frame, CONFIDENCE_THRESHOLD, self.NMS_THRESHOLD)
        return classes, scores, boxes
    
    def draw(self, frame,classes, scores, boxes):
        for (classid, score, box) in zip(classes, scores, boxes):
            color = COLORS[int(classid) % len(COLORS)]
            label = "%s : %f" % (self.class_names[classid[0]], score)
            cv2.rectangle(frame, box, color, 2)
            cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNN')
    parser.add_argument('modelfolder',      type=str,              help='model path')
    parser.add_argument('datafolder',     type=str,              help='data path')
    #parser.add_argument('threshold',     type=float,              help='data')
    args = parser.parse_args()
    inp = args.modelfolder
    dat = 0.4
    dnn_f = Dnn(inp,dat)
    vc = cv2.VideoCapture(os.path.join(datafolder,args.datafolder))
    
    while cv2.waitKey(1) & 0xFF != ord('q'):
        (grabbed, frame) = vc.read()
        if not grabbed:
            exit()


        start = time.time()
        classes, scores, boxes = dnn_f.detect(frame)
        end = time.time()

        start_drawing = time.time()
        dnn_f.draw(frame, classes, scores, boxes)
        end_drawing = time.time()
    
        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("detections", frame)
    cv2.destroyAllWindows()