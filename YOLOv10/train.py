from ultralytics import YOLO
import os
import logging

log_dir = "/blue/hulcr/eric.kuo/YOLOv10"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    model = YOLO("yolov10x.pt")
    logging.info("Loaded Model")
    
    logging.info("Start Training")
    results = model.train(data="data.yaml", device=[0,1], batch=32, epochs=300, workers=8, patience=0, cache='disk')
    logging.info("Training Completed")
    
if __name__ == '__main__':
    main()
else:
    print("Script unable to run")
