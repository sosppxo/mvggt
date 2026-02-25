import os
import numpy as np
import json
from tqdm import tqdm

def generate_invalid_list():
    # Update this path to your ScanNet color/pose directory
    data_root = 'data/scannet_data'
    scenes = sorted(os.listdir(data_root))
    invalid_list = {}

    for scene in tqdm(scenes, desc="Scanning scenes"):
        invalid_list[scene] = []
        pose_path = os.path.join(data_root, scene, 'pose')
        if not os.path.isdir(pose_path):
            continue
            
        pose_files = sorted(os.listdir(pose_path))
        for pose_file in pose_files:
            try:
                frame_id = int(os.path.splitext(pose_file)[0])
            except ValueError:
                continue # Skip non-numeric filenames

            with open(os.path.join(pose_path, pose_file), 'r') as f:
                try:
                    camera_pose = np.array([float(x) for x in f.read().split()]).astype(np.float32).reshape(4, 4)
                except (ValueError, IndexError):
                    # Handle cases where the file might be empty or malformed
                    invalid_list[scene].append(frame_id)
                    continue

            if np.isinf(camera_pose).any() or np.isnan(camera_pose).any():
                invalid_list[scene].append(frame_id)

    # Create a new dictionary with only the scenes that have invalid frames
    final_invalid_list = {scene: ids for scene, ids in invalid_list.items() if ids}

    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'scannet_invalid_list.json')

    with open(output_path, 'w') as f:
        json.dump(final_invalid_list, f, indent=4)
    
    print(f"Generated invalid list at {output_path}")
    print(f"Found {len(final_invalid_list)} scenes with invalid poses.")

if __name__ == '__main__':
    generate_invalid_list()