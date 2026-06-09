import os
import glob

def main():
    meta_lines = []

    # Parse Sintel
    sintel_root = 'data/MPI-Sintel/training/clean'
    if os.path.exists(sintel_root):
        for seq in sorted(os.listdir(sintel_root)):
            seq_path = os.path.join(sintel_root, seq)
            if not os.path.isdir(seq_path): continue
            
            frames = sorted(glob.glob(os.path.join(seq_path, '*.png')))
            if not frames: continue
            
            # frames are like frame_0001.png
            basename = os.path.basename(frames[0])
            prefix = basename.split('_')[0]
            indices = [int(os.path.splitext(os.path.basename(f))[0].split('_')[1]) for f in frames]
            
            min_idx = min(indices)
            max_idx = max(indices)
            num_frames = max_idx - min_idx + 1
            
            # sintel MPI-Sintel/training/clean/alley_1 frame 1 50 4 .png
            rel_path = f"MPI-Sintel/training/clean/{seq}"
            meta_lines.append(f"sintel {rel_path} {prefix} {min_idx} {max_idx} 4 .png")

    # Parse vKITTI
    vkitti_root = 'data/vKITTI'
    if os.path.exists(vkitti_root):
        for scene in sorted(os.listdir(vkitti_root)):
            scene_path = os.path.join(vkitti_root, scene)
            if not os.path.isdir(scene_path): continue
            
            for variation in sorted(os.listdir(scene_path)):
                rgb_dir = os.path.join(scene_path, variation, 'frames', 'rgb', 'Camera_0')
                if not os.path.isdir(rgb_dir): continue
                
                frames = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))
                if not frames: continue
                
                # frames are like rgb_00000.jpg
                basename = os.path.basename(frames[0])
                prefix = basename.split('_')[0]
                indices = [int(os.path.splitext(os.path.basename(f))[0].split('_')[1]) for f in frames]
                
                min_idx = min(indices)
                max_idx = max(indices)
                num_frames = max_idx - min_idx + 1
                
                # vkitti vKITTI/Scene01/15-deg-left/frames/rgb/Camera_0 rgb 0 446 5 .jpg
                rel_path = f"vKITTI/{scene}/{variation}/frames/rgb/Camera_0"
                meta_lines.append(f"vkitti {rel_path} {prefix} {min_idx} {max_idx} 5 .jpg")

    os.makedirs('RDG/basicsr/data/meta_info', exist_ok=True)
    with open('RDG/basicsr/data/meta_info/combined_train_meta_info.txt', 'w') as f:
        f.write('\n'.join(meta_lines))
    print(f"Generated {len(meta_lines)} sequences in meta info.")

if __name__ == '__main__':
    main()