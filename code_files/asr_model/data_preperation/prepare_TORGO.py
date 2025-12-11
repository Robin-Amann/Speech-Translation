from code_files.magic_strings import TORGO_ORIGINAL_PATH_LOCAL, TORGO_PATH_LOCAL
from pathlib import Path
import os
import shutil


def read_original_torgo_files():
    root = Path(TORGO_ORIGINAL_PATH_LOCAL)
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
                elif os.path.isdir(session_dir / "wav_arrayMic") :
                    audio_dir = session_dir / "wav_arrayMic"
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
                        "session": session_dir.name,
                        "prompt": prompt,
                        "wav_file_path": str(wav_file)
                    })
    return samples


def filter_torgo(dataset: list[dict[str, str]]) :
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


def analyze_torgo(torgo: list[dict[str, str]]) :
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


def save(torgo: list[dict[str, str]]) :
    target_dir = Path(TORGO_PATH_LOCAL)
    target_dir.mkdir(parents=True, exist_ok=True)

    for entry in torgo :
        id = Path(entry["wav_file_path"]).stem
        filestem = f"{entry["speaker_id"]}_{entry["session"].replace("_", "and")}_{id}"


        (target_dir / f"{filestem}.txt").write_text(entry["prompt"])
        shutil.copy2(entry["wav_file_path"], (target_dir / f"{filestem}.wav"))