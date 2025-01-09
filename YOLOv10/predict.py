from ultralytics import YOLO
import os
from PIL import Image
import logging

def get_most_recent_run(directory):
    # get train files
    files = [
        f for f in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, f)) and f.startswith('train')
    ]
    if not files:
        print("No train files found in the directory.")
        return None

    # Find the most recent file by creation time
    most_recent_file = max(files, key=lambda f: os.path.getctime(os.path.join(directory, f)))

    return most_recent_file

def create_predictions_folder(base_name, parent_directory):
    # Construct the initial folder path
    folder_path = os.path.join(parent_directory, base_name)
    count = 1

    # Increment folder name if it already exists
    while os.path.exists(folder_path):
        new_folder_path = os.path.join(parent_directory, f"{base_name}_{count}")
        if not os.path.exists(new_folder_path):
            os.rename(folder_path, new_folder_path)
            print(f"Renamed folder from {folder_path} to {new_folder_path}")
            return new_folder_path
        
        count += 1

    return None

def main():
    path = '/blue/hulcr/eric.kuo/YOLOv10/runs/detect'
    lastRun = get_most_recent_run(path)
    weights = f'{path}/{lastRun}/weights/best.pt'
    logging.info(f"Using weights from {weights}")
    print(f"Using weights from {weights}")
    
    model = YOLO(weights)
    logging.info("Loaded Model")
    
    #dataset_folder = '/blue/hulcr/eric.kuo/test_dataset/train'
    #images_folder = '/blue/hulcr/eric.kuo/test_dataset/train/images'
    dataset_folder = '/blue/hulcr/share/eric.kuo/Beetle_classifier/Data/00_Preprocessed_composite_images/train'
    images_folder = '/blue/hulcr/share/eric.kuo/Beetle_classifier/Data/00_Preprocessed_composite_images/train/images'
    
    previous_pred_folder = create_predictions_folder('labels', dataset_folder)
    logging.info(f"Previous labels moved to: {previous_pred_folder}")
    
    output_folder = dataset_folder + '/labels'
    os.makedirs(output_folder, exist_ok=True)
    logging.info(f"Created output folder: {output_folder}")
    
    logging.info("Start predictions on images")
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):  # Add other image extensions if needed
                image_path = os.path.join(root, file)
                output_subfolder = os.path.join(output_folder, os.path.relpath(root, images_folder))
                os.makedirs(output_subfolder, exist_ok=True)

                predictions = model(image_path, device=0, verbose=False)

                output_txt_path = os.path.join(output_subfolder, os.path.splitext(file)[0] + ".txt")
                with open(output_txt_path, 'w') as f:
                    for idx, prediction in enumerate(predictions[0].boxes.xywhn):
                        cls = int(predictions[0].boxes.cls[idx].item())
                        f.write(f"{cls} {prediction[0].item()} {prediction[1].item()} {prediction[2].item()} {prediction[3].item()}\n")
                    

if __name__ == '__main__':
    main()
else:
    print("Script unable to run")