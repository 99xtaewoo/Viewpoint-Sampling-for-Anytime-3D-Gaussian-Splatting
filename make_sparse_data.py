import os
import shutil

# Directory where the images are located
path = "/home/ssu25/TaewooKim/gaussian-splatting/sparse_tandt/truck"
source_directory = path + "/images"
source_camera_directory = path + "/sparse"
# Directory where the images will be moved to

target_path = "/home/ssu25/TaewooKim/gaussian-splatting/sparse_tandt/new"
target_directory = target_path + '/images'
target_camera_directory = target_path 


#move camera dir
shutil.copytree(source_camera_directory, target_camera_directory)

# Create target directory if it does not exist
if not os.path.exists(target_directory):
    os.makedirs(target_directory)

file_list = os.listdir(source_directory)

# Loop through each file in the directory

Num_data = len(file_list)
# print(file_list)

file_name = f"{0:06d}.jpg"
source_path = os.path.join(source_directory, file_name)

# print(source_path)
if os.path.exists(source_path):
    start = 0
else:
    start = 1 

ratio = 0.3 # 새로운 데이터 세트가 기존의 30퍼센트일 경우 (0.3)
step = int(Num_data * ratio)
print(step)

# Loop through the files in the source directory
for i in range(start, Num_data, 8):  # Start from 0, go up to 255, step by 8
    
    file_name = f"{i:06d}.jpg"
    source_path = os.path.join(source_directory, file_name)

    print(source_path)
    # Check if the file exists
    # if os.path.exists(source_path):
        # Move the file to the target directory
    shutil.copy(source_path, os.path.join(target_directory, file_name))

# Indicate completion
print("Images have been moved successfully.")
