import os
import glob
import re

def generate_meta_info(data_root, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        if not os.path.exists(data_root):
            return
        scenes = os.listdir(data_root)
        for scene in scenes:
            scene_path = os.path.join(data_root, scene)
            if not os.path.isdir(scene_path): continue
            spawns = os.listdir(scene_path)
            for spawn in spawns:
                spawn_path = os.path.join(scene_path, spawn)
                if not os.path.isdir(spawn_path): continue
                res_path = os.path.join(spawn_path, "1440p")
                if not os.path.isdir(res_path): continue
                
                frames = glob.glob(os.path.join(res_path, "*.rgb.png"))
                frame_count = len(frames)
                if frame_count == 0: continue
                
                clip_name = f"{scene}/{spawn}"
                f.write(f"{clip_name} {frame_count} 0 0\n")

def patch_dataset_script():
    dataset_file = "RDG/basicsr/data/render_dataset.py"
    with open(dataset_file, 'r') as f:
        content = f.read()

    # We need to change the get_img_path lambda to handle GameIR paths
    # GameIR HR (GT): static_town08/19/1440p/00000000.rgb.png
    # GameIR LR (LQ): static_town08/19/720p/00000000.rgb.png
    # buffer types: 'Image' -> '.rgb.png', 'Depth' -> '.depth.png'
    
    # Original get_img_path in RenderRecurrentDataset
    # get_img_path = lambda root, buffer, ext: osp.join(root, clip_name, buffer, frame_str + patch_idx + ext)
    
    replacement = """
            # GameIR path formatting
            # frames are incremented by 10, e.g. 00000000, 00000010
            frame_idx_str = f"{i*10:08d}"
            
            def get_img_path(root, buffer, ext):
                res_folder = "1440p" if "1440p" in root or getattr(self, "is_gt_root", True) else "720p"
                # For training, let's just assume root has GameIR-SR structure
                # GT uses 1440p, LQ uses 720p
                if "gt" in root.lower() or "GT" in root:
                    res_folder = "1440p"
                elif "lq" in root.lower() or "LQ" in root:
                    res_folder = "720p"
                else:
                    # heuristic
                    res_folder = "1440p" if buffer == 'gt' else "720p"
                    
                if buffer == 'Image':
                    suffix = '.rgb.png'
                elif buffer == 'Depth':
                    suffix = '.depth.png'
                else:
                    suffix = ext
                
                return osp.join(root, clip_name, res_folder, frame_idx_str + suffix)
"""

    content = re.sub(
        r"            frame_str = f\"\{i:03d\}\"\n            get_img_path = lambda root, buffer, ext: osp.join\(root, clip_name, buffer, frame_str \+ patch_idx \+ ext\)",
        replacement,
        content
    )

    # Change load_exr to load_image for Depth in RenderRecurrentDataset
    content = content.replace(
        "lqs[buffer_type].append(load_exr(get_img_path(self.lq_root, buffer_type, '.exr'), buffer_type))",
        "lqs[buffer_type].append(load_image(get_img_path(self.lq_root, buffer_type, '.png'), self.format) if buffer_type == 'Depth' else load_exr(get_img_path(self.lq_root, buffer_type, '.exr'), buffer_type))"
    )
    content = content.replace(
        "gts[buffer_type].append(load_exr(get_img_path(self.gt_root, buffer_type, '.exr'), buffer_type))",
        "gts[buffer_type].append(load_image(get_img_path(self.gt_root, buffer_type, '.png'), self.format) if buffer_type == 'Depth' else load_exr(get_img_path(self.gt_root, buffer_type, '.exr'), buffer_type))"
    )

    # Do the same for RenderRecurrentTestDataset
    # Original:
    # img_main_path = osp.join(root_path, scene, 'Image', f"{image_name}.png")
    test_replacement = """
        res_folder = "1440p" if "gt" in image_type_prefix else "720p"
        # image_name is '000', convert to '00000000'
        frame_idx_str = f"{int(image_name)*10:08d}"
        img_main_path = osp.join(root_path, scene, res_folder, f"{frame_idx_str}.rgb.png")
"""
    content = re.sub(
        r"        img_main_path = osp.join\(root_path, scene, 'Image', f\"\{image_name\}\.png\"\)",
        test_replacement,
        content
    )
    
    # Original:
    # buffer_path = osp.join(root_path, scene, buffer_type, f"{image_name}.exr")
    test_buffer_replacement = """
                if buffer_type == 'Depth':
                    buffer_path = osp.join(root_path, scene, res_folder, f"{frame_idx_str}.depth.png")
                else:
                    buffer_path = osp.join(root_path, scene, res_folder, f"{frame_idx_str}.exr")
"""
    content = re.sub(
        r"                buffer_path = osp.join\(root_path, scene, buffer_type, f\"\{image_name\}\.exr\"\)",
        test_buffer_replacement,
        content
    )

    with open(dataset_file, 'w') as f:
        f.write(content)

def patch_yaml_config():
    config_file = "RDG/options/train/train_RDG-s_x4.yml"
    with open(config_file, 'r') as f:
        content = f.read()

    # Disable normal, brdf, motion as GameIR doesn't have them in this format
    content = content.replace("use_lq_normal: true", "use_lq_normal: false")
    content = content.replace("use_lq_brdf: true", "use_lq_brdf: false")
    content = content.replace("use_lq_motion: true", "use_lq_motion: false")
    content = content.replace("use_gt_normal: true", "use_gt_normal: false")
    content = content.replace("use_gt_brdf: true", "use_gt_brdf: false")
    content = content.replace("use_gt_motion: true", "use_gt_motion: false")

    with open(config_file, 'w') as f:
        f.write(content)

if __name__ == "__main__":
    print("Generating Meta Info...")
    generate_meta_info("GameIR/mini_dataset/train/GameIR-SR/GameIR-SR", "RDG/basicsr/data/meta_info/TrainData_meta_info.txt")
    generate_meta_info("GameIR/mini_dataset/test/GameIR-SR/GameIR-SR", "RDG/basicsr/data/meta_info/ValData_meta_info.txt")
    
    print("Patching dataset script...")
    patch_dataset_script()
    
    print("Patching yaml config...")
    patch_yaml_config()
    print("Environment prepared.")
