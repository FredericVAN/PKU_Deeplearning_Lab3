import os
import subprocess
import argparse

def download_data():
    """
    Download SVHN dataset if not present
    """
    print("Downloading SVHN dataset...")
    if not os.path.exists('train.tar.gz'):
        subprocess.run(['wget', 'http://ufldl.stanford.edu/housenumbers/train.tar.gz'])
    if not os.path.exists('test.tar.gz'):
        subprocess.run(['wget', 'http://ufldl.stanford.edu/housenumbers/test.tar.gz'])
    
    # Extract if directories don't exist
    if not os.path.exists('train'):
        print("Extracting training data...")
        subprocess.run(['tar', '-xzf', 'train.tar.gz'])
    if not os.path.exists('test'):
        print("Extracting test data...")
        subprocess.run(['tar', '-xzf', 'test.tar.gz'])

def run_pipeline(skip_data_prep=False, skip_training=False, skip_eval=False):
    """
    Run the entire pipeline
    """
    # Download and extract data
    download_data()
    
    # Prepare data for YOLO
    if not skip_data_prep:
        print("Preparing data for YOLO format...")
        subprocess.run(['python', 'prepare_data.py'])
    
    # Train the model
    if not skip_training:
        print("Training YOLO model...")
        subprocess.run(['python', 'train.py'])
    
    # Evaluate the model
    if not skip_eval:
        print("Evaluating model...")
        subprocess.run(['python', 'evaluate.py'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SVHN Recognition Pipeline')
    parser.add_argument('--skip_data_prep', action='store_true', help='Skip data preparation step')
    parser.add_argument('--skip_training', action='store_true', help='Skip model training step')
    parser.add_argument('--skip_eval', action='store_true', help='Skip evaluation step')
    
    args = parser.parse_args()
    
    run_pipeline(
        skip_data_prep=args.skip_data_prep,
        skip_training=args.skip_training,
        skip_eval=args.skip_eval
    )