import os
import h5py
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm

def create_yolo_dataset(data_path, output_path, split='train'):
    """
    Convert SVHN dataset to YOLO format
    """
    # Create necessary directories
    os.makedirs(os.path.join(output_path, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'labels', split), exist_ok=True)
    
    # Read digitStruct.mat
    with h5py.File(os.path.join(data_path, 'digitStruct.mat'), 'r') as h5f:
        digit_struct = h5f['digitStruct']
        bbox_dataset = digit_struct['bbox']
        name_dataset = digit_struct['name']
        
        for idx in tqdm(range(len(bbox_dataset))):
            # Get image name
            name_ref = name_dataset[idx][0]
            name_obj = h5f[name_ref]
            img_name = ''.join(chr(val[0]) for val in name_obj[()])
            
            # Copy image to YOLO format directory
            src_img_path = os.path.join(data_path, img_name)
            dst_img_path = os.path.join(output_path, 'images', split, img_name)
            shutil.copy2(src_img_path, dst_img_path)
            
            # Get bbox information
            bbox_ref = bbox_dataset[idx][0]
            bbox_obj = h5f[bbox_ref]
            
            # Get attributes
            label_refs = bbox_obj['label'][()]
            left_refs = bbox_obj['left'][()]
            top_refs = bbox_obj['top'][()]
            width_refs = bbox_obj['width'][()]
            height_refs = bbox_obj['height'][()]
            
            # Open image to get dimensions
            img = Image.open(src_img_path)
            img_width, img_height = img.size
            
            # Create YOLO format labels
            yolo_labels = []
            
            for i in range(len(label_refs)):
                # Get digit value - 修复HDF5引用对象处理
                label_ref = label_refs[i][0]
                if isinstance(label_ref, h5py.h5r.Reference):
                    digit_value = int(h5f[label_ref][()][0])
                else:
                    digit_value = int(label_ref)
                
                # Get bounding box coordinates - 修复HDF5引用对象处理
                left_ref = left_refs[i][0]
                top_ref = top_refs[i][0]
                width_ref = width_refs[i][0]
                height_ref = height_refs[i][0]
                
                if isinstance(left_ref, h5py.h5r.Reference):
                    x = float(h5f[left_ref][()][0])
                else:
                    x = float(left_ref)
                
                if isinstance(top_ref, h5py.h5r.Reference):
                    y = float(h5f[top_ref][()][0])
                else:
                    y = float(top_ref)
                
                if isinstance(width_ref, h5py.h5r.Reference):
                    w = float(h5f[width_ref][()][0])
                else:
                    w = float(width_ref)
                
                if isinstance(height_ref, h5py.h5r.Reference):
                    h = float(h5f[height_ref][()][0])
                else:
                    h = float(height_ref)
                
                # Convert to YOLO format (normalized coordinates)
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                
                # YOLO format: <class> <x_center> <y_center> <width> <height>
                yolo_labels.append(f"{digit_value} {x_center} {y_center} {width} {height}")
            
            # Save labels
            label_path = os.path.join(output_path, 'labels', split, 
                                     os.path.splitext(img_name)[0] + '.txt')
            with open(label_path, 'w') as txtf:
                txtf.write('\n'.join(yolo_labels))

def main():
    # Create dataset.yaml with correct YAML format
    dataset_yaml = """
    path: D://mycoding/python/PKU_Deeplearning_Lab3/dataset  # 使用完整的绝对路径
    train: D://mycoding/python/PKU_Deeplearning_Lab3/dataset/images/train  # 完整的训练集路径
    val: D://mycoding/python/PKU_Deeplearning_Lab3/dataset/images/test    # 完整的验证集路径

    # Classes
    names:
        0: 'Nan'
        1: '1'
        2: '2'
        3: '3'
        4: '4'
        5: '5'
        6: '6'
        7: '7'
        8: '8'
        9: '9'
        10: '0' #共 10 个类别，每个数字对应 1 个。数字“1”的标签为 1，“9”的标签为 9，“0”的标签为 10。
    """
    
    # Create dataset directory
    os.makedirs('dataset', exist_ok=True)
    
    # Save dataset.yaml
    with open('dataset/dataset.yaml', 'w') as f:
        f.write(dataset_yaml)
    
    # Download data if not exist
    if not os.path.exists('train') or not os.path.exists('test'):
        print("请从以下链接下载并解压SVHN数据集:")
        print("http://ufldl.stanford.edu/housenumbers/")
        print("将train.tar.gz解压到'train/'目录，将test.tar.gz解压到'test/'目录")
        return
    
    # Process train and test datasets
    print("处理训练数据...")
    create_yolo_dataset('train', 'dataset', 'train')
    
    print("处理测试数据...")
    create_yolo_dataset('test', 'dataset', 'test')

if __name__ == '__main__':
    main()