import os
import shutil

def clean_directories():
    directories = ['results', 'states', 'weights']
    
    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
            print(f"Cleaned and recreated directory: {directory}")
        else:
            os.makedirs(directory)
            print(f"Created directory: {directory}")

if __name__ == "__main__":
    clean_directories()