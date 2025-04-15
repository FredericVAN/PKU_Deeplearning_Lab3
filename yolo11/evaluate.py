from ultralytics import YOLO
import os
import numpy as np
from tqdm import tqdm

def evaluate_model():
    # Load the trained model
    model = YOLO('runs/detect/svhn_model/weights/best.pt')
    
    # Run inference on the test set
    results = model.predict(
        source='dataset/images/test',
        save=True,
        save_txt=True,
        conf=0.25
    )
    
    # Calculate accuracy
    correct = 0
    total = 0
    
    for result in tqdm(results):
        # Get predicted digits
        pred_digits = []
        for box in result.boxes:
            pred_digits.append(int(box.cls.item()))
        
        # Sort digits from left to right based on x-coordinate
        if pred_digits:
            # Get x-coordinates for each detected digit
            x_coords = [box.xyxy.cpu().numpy()[0][0] for box in result.boxes]  # x-min values
            # Sort digits based on x-coordinates (left to right)
            pred_digits = [d for _, d in sorted(zip(x_coords, pred_digits))]
        
        # Get ground truth from label file
        gt_digits = []
        img_path = result.path
        label_path = img_path.replace('images', 'labels').replace('.png', '.txt')
        
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                
                # Extract label and x-coordinate for each digit
                label_x_pairs = []
                for line in lines:
                    parts = line.strip().split()
                    label = int(parts[0])
                    x_center = float(parts[1])
                    label_x_pairs.append((x_center, label))
                
                # Sort by x-coordinate (left to right)
                label_x_pairs.sort()
                gt_digits = [label for _, label in label_x_pairs]
            
            # Compare predictions with ground truth
            if pred_digits and len(pred_digits) == len(gt_digits):
                # Check if all digits match
                if ''.join(map(str, pred_digits)) == ''.join(map(str, gt_digits)):
                    correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Total samples: {total}")
    print(f"Correctly recognized: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return accuracy

if __name__ == '__main__':
    evaluate_model()