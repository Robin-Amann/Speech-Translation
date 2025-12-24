import torchaudio
import os
import ast
from pathlib import Path
from torch import Tensor

def read_audio(file_path: str | Path, sample_rate: int) :
    if not os.path.isfile(file_path) :
        print('audiofile does not exist', file_path)
        return []
    waveform, _ = torchaudio.load(file_path)  
    if torchaudio.info(file_path).sample_rate != sample_rate :    
        waveform = torchaudio.functional.resample(
            orig_freq=torchaudio.info(file_path).sample_rate, 
            new_freq=sample_rate, 
            waveform=waveform)
    return waveform

    
def write_audio(file_path: str | Path, waveform: Tensor, sample_rate: int) :
    if waveform.dim() == 1 :
        waveform = waveform[None, :]     # [a, b, c] -> [[a, b, c]] 
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torchaudio.save(file_path, waveform, sample_rate)


def read_file(file_path) :
    content = ""
    with open(file_path, "r", encoding="utf8") as file :
        content = file.read()
    return content


def write_file(file_path: str | Path, content: str, mode='w') :
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode, encoding='utf8') as file :
        file.write(content)


def read_dict(file_path: str | Path, seperator='<|>') :
    """reads a list of dicts from a file. all dicts need to have the same keys.\\
    key1<|>...<|>keyN\\
    type1<|>...<|>typeN\\
    values of dict 1\\
    ..."""
    dictionary = []
    lines = read_file(file_path)
    if not lines :
        return dictionary
    lines = lines.split('\n')
    if len(lines) == 2 :
        raise KeyError('file probably had no header and only two items')
    keys, types = lines[:2]
    keys = keys.split(seperator)    
    types = types.split(seperator)
    if len(keys) != len(types) :
        raise KeyError('wrong number of keys to types')

    data = lines[2:]
    for d in data :
        current_dict = dict()
        for k, t, v in zip(keys, types, d.split(seperator)) :
            if t == "str" :
                current_dict[k] = v
            else :
                current_dict[k] = ast.literal_eval(v)
        dictionary.append(current_dict)
    return dictionary


def write_dict(file_path: str | Path, data_p: list[dict], separator='<|>') :
    """writes a list of dicts to a file. all dicts need to have the same keys.\\
    key1<|>...<|>keyN\\
    type1<|>...<|>typeN\\
    values of dict 1\\
    ..."""
    data = data_p.copy()
    if type(data) != list or not all(type(item) == dict for item in data) :
        return
    if len(data) == 0 :
        write_file(file_path, '')
        return
    keys = list(data[0].keys())
    types = [type(data[0][key]).__name__ for key in keys ]
    types = [ t if t != int else float for t in types ]
    lines = [separator.join(keys), separator.join(types)]
    for value in data :
        line = []
        for key in keys :
            line.append(str(value[key]))
        lines.append( separator.join(line) )
    write_file(file_path, '\n'.join(lines))