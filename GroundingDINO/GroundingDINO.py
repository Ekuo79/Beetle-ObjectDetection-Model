import os
import logging
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

Image.MAX_IMAGE_PIXELS = None

log_dir = "/blue/hulcr/eric.kuo"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    logging.info("Starting to load model...")
    model_id = "IDEA-Research/grounding-dino-base"
    #model_id = "IDEA-Research/grounding-dino-tiny"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)
    logging.info("Model loaded.")

    dataset_folder = '/blue/hulcr/share/eric.kuo/Beetle_classifier/Data/00_Preprocessed_composite_images/train'
    output_folder = '/blue/hulcr/share/eric.kuo/Beetle_classifier/Data/GroundingDINO_1'
    #dataset_folder = '/blue/hulcr/eric.kuo/test_dataset'
    #output_folder = '/blue/hulcr/eric.kuo/test_output'
    
    text_queries = 'a bug.'

    os.makedirs(output_folder, exist_ok=True)

    # Function to convert bounding boxes to YOLO format
    def convert_to_yolo_format(image_size, boxes):
        yolo_boxes = []
        for box in boxes:
            x_min, y_min, x_max, y_max = box
            width = x_max - x_min
            height = y_max - y_min
            x_center = (x_min + x_max) / 2.0
            y_center = (y_min + y_max) / 2.0

            # Normalize coordinates
            x_center /= image_size[0]
            y_center /= image_size[1]
            width /= image_size[0]
            height /= image_size[1]

            yolo_boxes.append(f"0 {x_center} {y_center} {width} {height}")  # Assuming class index 0 for beetles

        return yolo_boxes
    
    logging.info("Starting to process directory...")
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):  # Add other image extensions if needed
                image_path = os.path.join(root, file)
                output_subfolder = os.path.join(output_folder, os.path.relpath(root, dataset_folder))
                os.makedirs(output_subfolder, exist_ok=True)

                output_txt_path = os.path.join(output_subfolder, os.path.splitext(file)[0] + ".txt")
                if os.path.exists(output_txt_path):
                    continue

                # Load image
                image = Image.open(image_path).convert("RGB")

                # Process image with Grounding DINO
                inputs = processor(images=image, text=text_queries, return_tensors="pt").to(DEVICE)

                with torch.no_grad():
                    outputs = model(**inputs)

                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=0.4,
                    text_threshold=0.6,
                    target_sizes=[image.size[::-1]]
                )

                # Convert and save annotations in YOLO format
                image_size = image.size
                yolo_annotations = convert_to_yolo_format(image_size, results[0]['boxes'])

                # Save annotations to a txt file
                with open(output_txt_path, 'w') as f:
                    for annotation in yolo_annotations:
                        f.write(annotation + '\n')
    logging.info("Directory processing completed.")

    logging.info("---------DONE---------")

if __name__ == '__main__':
    main()
else:
    print("Script unable to run")
