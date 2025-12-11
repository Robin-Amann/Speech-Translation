import os
import code_files.magic_strings as magic_string
from pathlib import Path
from datasets import Dataset

def print_directory_tree(root_path: str, prefix="", depth=0):
    if depth == 0 :
        if root_path.endswith("/") :
            root_path = root_path[:-1]

        print(root_path.split("/")[-1])

    try:
        entries = sorted(
            [e for e in os.listdir(root_path)
             if os.path.isdir(os.path.join(root_path, e))]
        )
    except OSError as e:
        print(f"Error accessing '{root_path}': {e}")
        return

    for i, entry in enumerate(entries):
        connector = "└─ " if i == len(entries) - 1 else "├─ "
        print(prefix + connector + entry)
        next_prefix = prefix + ("   " if i == len(entries) - 1 else "│  ")
        print_directory_tree(os.path.join(root_path, entry), next_prefix, depth=depth+1)


def print_all_prompts(root):
    i = 0
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == "prompts":
            # print(f"\n=== Directory: {dirpath} ===\n")
            for fname in sorted(filenames):
                fpath = os.path.join(dirpath, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                    # print(f"--- {fname} ---")
                    if "[" in content and "]" in content :
                        continue
                    if " " in content:
                        print(fname.ljust(30), content)
                        i += 1
                    # print()
                except Exception as e:
                    print(f"Could not read {fpath}: {e}")
    return i


def load_torgo(local: bool=True) :
    if local :
        target_dir = Path(magic_string.TORGO_PATH_LOCAL)
    else :
        raise NotImplementedError("not implemented")

    dataset = []
    for txt_file in target_dir.glob("*.txt"):  # only files ending with .txt, non-recursive
        speaker_id, session, id = txt_file.stem.split("_")
        prompt = txt_file.read_text(encoding="utf-8")
        wav_file = txt_file.with_suffix(".wav")
        dataset.append({
            "speaker_id": speaker_id,
            "session": session,
            "id": id,
            "prompt": prompt,
            "wav_file_path": str(wav_file)
        })

    return Dataset.from_list(dataset)

def save_embeddings(dataset: list[dict]) :
    target_dir = Path(magic_string.TORGO_EMBEDDINGS_PATH_LOCAL)
    target_dir.mkdir(parents=True, exist_ok=True)

    for entry in dataset :
        if "embedding" not in entry :
            continue
        file = target_dir / f"{Path(entry["wav_file_path"]).stem}.txt"
        file.write_text(str(entry["embedding"])[1:-1])

def load_embeddings(dataset: list[dict]) :
    target_dir = Path(magic_string.TORGO_EMBEDDINGS_PATH_LOCAL)

    for entry in dataset :
        if "embedding" in entry :
            continue

        file = target_dir / f"{Path(entry["wav_file_path"]).stem}.txt"
        with open(file, "r", encoding="utf-8") as f:
            embedding = f.read().strip()
        entry["embedding"] = [ float(x) for x in embedding.split(", ") ]

    return dataset



# print_directory_tree("./data/datasets/TORGO")
# i = print_all_prompts("./data/datasets/TORGO")
# print(i)