from huggingface_hub import dataset_info
info = dataset_info("LLLebin/GameIR")
print("Total size to download (bytes):", info.siblings)
