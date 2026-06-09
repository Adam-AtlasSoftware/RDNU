import os
import glob

extracted_dir = "/home/adam/2tb-workspace/RDNU/M3VIR-extracted"
meta_file = "/home/adam/2tb-workspace/RDNU/RDG/basicsr/data/meta_info/m3vir_train_meta_info.txt"

# Find all RGB sequences
search_path = os.path.join(extracted_dir, "**", "RGB_images")
rgb_dirs = glob.glob(search_path, recursive=True)

valid_count = 0

with open(meta_file, 'w') as f:
    for rgb_dir in rgb_dirs:
        png_files = glob.glob(os.path.join(rgb_dir, "**", "*.png"), recursive=True)
        if not png_files:
            continue
        
        # Group by prefix
        prefixes = {}
        for p in png_files:
            basename = os.path.basename(p).replace(".png", "")
            parts = basename.split("_")
            if len(parts) > 1 and parts[-1].isdigit():
                prefix = "_".join(parts[:-1])
                idx = int(parts[-1])
                if prefix not in prefixes:
                    prefixes[prefix] = []
                # Also ensure the file isn't empty!
                if os.path.getsize(p) > 0:
                    prefixes[prefix].append((idx, p))
        
        for prefix, frames in prefixes.items():
            frames.sort(key=lambda x: x[0])
            
            # Find continuous chunks
            if not frames:
                continue
                
            chunks = []
            current_chunk = [frames[0]]
            for i in range(1, len(frames)):
                if frames[i][0] == current_chunk[-1][0] + 1:
                    current_chunk.append(frames[i])
                else:
                    if len(current_chunk) >= 6:
                        chunks.append(current_chunk)
                    current_chunk = [frames[i]]
            if len(current_chunk) >= 6:
                chunks.append(current_chunk)
                
            for chunk in chunks:
                min_idx = chunk[0][0]
                max_idx = chunk[-1][0]
                sample_path = os.path.dirname(chunk[0][1])
                rel_path = os.path.relpath(sample_path, extracted_dir)
                f.write(f"{rel_path} {prefix} {len(chunk)} {min_idx} {max_idx}\n")
                valid_count += 1

print(f"Meta info generated at {meta_file} with {valid_count} continuous sequences.")
