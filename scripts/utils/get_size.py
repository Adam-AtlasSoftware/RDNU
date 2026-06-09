from huggingface_hub import dataset_info
info = dataset_info("LLLebin/GameIR")
total_size = sum(f.size for f in info.siblings if getattr(f, 'size', None))
if total_size == 0: # sizes might be in a different attribute or need another API
    print("Could not retrieve sizes directly. Siblings count:", len(info.siblings))
else:
    print(f"Total size: {total_size / (1024**3):.2f} GB")
