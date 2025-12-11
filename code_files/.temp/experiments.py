import os

def print_tree(path: str, prefix: str = ""):
    """Recursively prints the directory structure in tree form."""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory")

    entries = sorted(os.listdir(path))
    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(full_path):
            extension = "    " if i == len(entries) - 1 else "│   "
            print_tree(full_path, prefix + extension)


# Example usage:
# print_tree("/path/to/directory")


print_tree("./asr_model/checkpoints")

# checkpoints
# ├── .locks                                        Contains lock files used to prevent parallel downloads from corrupting the cache     
# │   └── models--openai--whisper-small                 
# └── models--openai--whisper-small                 cached copy of the model
#     ├── .no_exist                                 You attempted to access a file that does not exist in the repository
#     │   └── 973afd24965f72e36ca33b3055d56a652f456b4d
#     │       ├── adapter_config.json
#     │       └── custom_generate
#     │           └── generate.py
#     ├── blobs                                     Stores the actual binary blobs that were downloaded
#     ├── refs                                      Contains a single text file storing the commit hash of a reference, typically a branch like main
#     │   └── main
#     └── snapshots                                 A snapshot is a full, immutable local copy of a specific Git commit of the model repository
#         └── 973afd24965f72e36ca33b3055d56a652f456b4d
#             ├── config.json
#             ├── generation_config.json
#             └── model.safetensors