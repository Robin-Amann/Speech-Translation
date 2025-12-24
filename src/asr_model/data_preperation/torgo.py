import src.magic_strings as magic_strings
from pathlib import Path
import os
import shutil
from pathlib import Path

# file content:
# speaker_id
# session
# id
# prompt
# wav_file_path
# (embedding)
# (cohort)

# read_original_torgo_files
def load_original():
    root = magic_strings.TORGO_PATH_LOCAL / "original"
    samples = []

    # iterate over gender folders (F, M)
    for gender_dir in root.iterdir():
        # iterate over speaker dirs (F01, M02, ...)
        for speaker_dir in gender_dir.iterdir():
            # iterate over sessions (Session1, Session2, ...)
            for session_dir in speaker_dir.iterdir():
                if session_dir.is_file() or not session_dir.name.startswith("Session") :
                    continue
                prompts_dir = session_dir / "prompts"

                # find the audio directory in this session
                if os.path.isdir(session_dir / "wav_headMic") :
                    audio_dir = session_dir / "wav_headMic"
                    source = "headMic"
                elif os.path.isdir(session_dir / "wav_arrayMic") :
                    audio_dir = session_dir / "wav_arrayMic"
                    source = "arrayMic"
                else :
                    raise ValueError(f"no wav directory ({session_dir})")

                # iterate over all prompt txt files
                for prompt_file in prompts_dir.glob("*.txt"):
                    stem = prompt_file.stem
                    wav_file = audio_dir / f"{stem}.wav"

                    with open(prompt_file, "r", encoding="utf-8") as f:
                        prompt = f.read().strip()

                    if not os.path.isfile(wav_file) :
                        print(f"prompt file has no wav file ({prompt_file})")
                        print(f"- content: {prompt}")
                        continue

                    samples.append({
                        "speaker_id": speaker_dir.name,
                        "session": session_dir.name.replace("_", "and"),
                        "id": stem,
                        "prompt": prompt,
                        "wav_file_path": str(wav_file),
                        "source": source
                    })
    return samples


def filter_original(dataset: list[dict[str, str]]) :
    sentences = []
    for entry in dataset :
        # prompt is image
        if "/" in entry["prompt"] :
            # print(file["prompt"])
            continue

        # prompt is command
        # e.g. 
        # - [relax your mouth in its normal position]
        # - tear [as in tear up that paper]
        if "[" in entry["prompt"] and "]" in entry["prompt"] :
            # print(file["prompt"])
            continue
        
        # prompt is single word
        if " " not in entry["prompt"] :
            # print(file["prompt"])
            continue
        
        if " " in entry["prompt"] :
            sentences.append(entry)
            continue

        raise ValueError(f"unexpected prompt: {entry["prompt"]}")

    return sentences


def analyze_original(torgo: list[dict[str, str]]) :
    from collections import defaultdict

    grouped : dict[str, list] = {}

    for item in torgo:
        if item["speaker_id"] not in grouped :
            grouped[item["speaker_id"]] = []
        grouped[item["speaker_id"]].append(item)

    print("speaker", "#sentences")
    for speaker, sentences in grouped.items() :
        print(speaker.ljust(7), len(sentences))

    print("sum".ljust(7), len(torgo))


def save_file(split, entry) :
    """use together with utils.split_and_save_dataset"""
    cohort = entry["cohort"] if "cohort" in entry else ""
    if split == None :
        split = ""
    target_dir = Path(magic_strings.TORGO_PATH_LOCAL / cohort / entry["speaker_id"] / split)
    target_dir.mkdir(parents=True, exist_ok=True)

    id = Path(entry["wav_file_path"]).stem.split("_")[-1]
    stem = f"{entry["speaker_id"]}_{entry["session"]}_{id}"

    (target_dir / f"{stem}.txt").write_text(entry["prompt"])
    if not (target_dir / f"{stem}.wav").is_file() :
        shutil.copy2(entry["wav_file_path"], (target_dir / f"{stem}.wav"))

    if "cohort" in entry :
        (target_dir / f"{stem}.ch").write_text(entry["cohort"])
    if "embedding" in entry :
        (target_dir / f"{stem}.emb").write_text(str(entry["embedding"])[1:-1]) # remove [ and ]
        

def load_file(path: Path) :
    if path.suffix != ".txt" :
        return None
    
    speaker_id, session, id = path.stem.split("_")
    split = path.parent.name
    if split not in ["train", "dev", "test"] :
        split = None

    entry = {
        "speaker_id": speaker_id,
        "session": session,
        "id": id,
        "prompt": path.read_text(encoding="utf-8"),
        "wav_file_path": str(path.with_suffix(".wav"))
    }

    if path.with_suffix(".ch").is_file() :
        entry["cohort"] = path.with_suffix(".ch").read_text(encoding="utf-8")
    if path.with_suffix(".emb").is_file() :
        emb = path.with_suffix(".emb").read_text(encoding="utf-8").strip().split(", ")
        entry["embedding"] = [ float(x) for x in emb ]
        
    return split, entry