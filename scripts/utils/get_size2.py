from huggingface_hub import HfApi
api = HfApi()
info = api.dataset_info("LLLebin/GameIR", files_metadata=True)
total_size = sum(f.size for f in info.siblings if getattr(f, 'size', None) is not None)
print(f"Total size: {total_size / (1024**3):.2f} GB")
