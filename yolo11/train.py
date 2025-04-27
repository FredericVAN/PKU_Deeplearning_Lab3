import torch
from ultralytics import YOLO
import os

def see_chaocanshu():
    model = YOLO('yolo11n.pt')
    # 打印所有默认配置
    print(model.model.args)
    
def train_model():
    # Check if CUDA is available and set device accordingly
    if torch.cuda.is_available():
        device = '0'  # Use GPU
        print(f"使用GPU训练: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("CUDA不可用，使用CPU训练")
    
    # Load a model
    model = YOLO('yolo11n.pt')  # 加载预训练模型
    
    # Train the model
    results = model.train(
        data=r'D:\mycoding\python\PKU_Deeplearning_Lab3\dataset\dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='svhn_model',
        patience=20,
        save=True,
        device=device
    )
    
    return results

if __name__ == '__main__':
    see_chaocanshu()
    # train_model()