import os
from huggingface_hub import snapshot_download

# Download the GameIR dataset
print("Starting download of LLLebin/GameIR...")
snapshot_download(
    repo_id="LLLebin/GameIR",
    repo_type="dataset",
    local_dir="/home/adam/workspace/RDNU/GameIR",
    max_workers=8
)
print("Download complete.")