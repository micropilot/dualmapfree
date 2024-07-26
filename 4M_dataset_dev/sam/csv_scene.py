import glob
import csv
import os

train_scenes = glob.glob("/home/ubuntu/MapFree/data/original/train/**")
val_scenes = glob.glob("/home/ubuntu/MapFree/data/original/val/*")
scenes = train_scenes + val_scenes
num_csv = 4
scenes_per_csv = len(scenes) // num_csv

output_dir = "csv"
os.makedirs(output_dir, exist_ok=True)

for i in range(num_csv):
    start_index = i * scenes_per_csv
    if i == num_csv - 1:  
        end_index = len(scenes)
    else:
        end_index = (i + 1) * scenes_per_csv
    
    csv_filename = os.path.join(output_dir, f"scenes_part_{i}.csv")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['scene'])
        for scene in scenes[start_index:end_index]:
            writer.writerow([scene])
