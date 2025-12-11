from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import code_files.magic_strings as magic_strings
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from numpy.typing import NDArray
from typing import Literal
from datasets import Dataset
import torch
import torchaudio
import umap
import matplotlib.pyplot as plt
from pathlib import Path

type CovarianceType = Literal['full', 'tied', 'diag', 'spherical']

def fit_GMM(embeddings: list[list[int]], n_components: int, covariance_type: CovarianceType='full') :

    embeddings_array = np.asarray(embeddings)

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    gmm.fit(embeddings_array)

    labels = gmm.predict(embeddings_array)
    return gmm, labels


def fit_DPGMM(embeddings: list[list[int]], max_components=100, covariance_type: CovarianceType='full'):
    embeddings_array: NDArray = np.asarray(embeddings)

    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type='dirichlet_process',
        max_iter=500,
        init_params='kmeans'
    )

    model.fit(embeddings_array)    
    labels = model.predict(embeddings_array)
    
    return model, labels


def generate_embeddings(dataset: Dataset, sampling_rate: int=16000, batch_size=8) :

    # you need to use the same speech encoder that you use for the hypermodel
    feature_extractor = WhisperFeatureExtractor.from_pretrained(magic_strings.WHISPER_V3_MODEL_NAME, cache_dir=magic_strings.WHISPER_V3_CHECKPOINT_DIR)
    speech_encoder = WhisperModel.from_pretrained(magic_strings.WHISPER_V3_MODEL_NAME, cache_dir=magic_strings.WHISPER_V3_CHECKPOINT_DIR).encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    speech_encoder.to(device)
    speech_encoder.eval()

    if "audio" not in dataset.column_names :
        def read_audio(item) :
            waveform, sr = torchaudio.load(item["wav_file_path"])  # waveform shape: (channels, time)
            waveform = waveform.mean(dim=0)       # convert to mono if stereo
            if sr != sampling_rate:
                waveform = torchaudio.functional.resample(waveform, sr, sampling_rate)
            item["audio"] = waveform.numpy()
            return item
        
        dataset = dataset.map(read_audio)
        
    def generate_embeddings_vectorized(batch):
        # Convert all audios to tensors

        audios = [ a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else a for a in batch["audio"] ]
        
        # Extract features in batch
        inputs = feature_extractor(audios, sampling_rate=sampling_rate, return_tensors="pt") # padding=True
        input_features = inputs["input_features"].to(device)

        with torch.no_grad():
            encoder_outputs = speech_encoder(input_features) # BaseModelOutput
            if hasattr(encoder_outputs, "last_hidden_state"):
                hidden_states = encoder_outputs.last_hidden_state
            else:
                hidden_states = encoder_outputs[0]
            # Average over time dimension (dim=1)
            embeddings = hidden_states.mean(dim=1) # dim 1 because dim 0 is batch and dim 1 is time
            # model.config.d_model should equal len(embedding). (whisper-v3 1280)

        # Convert embeddings to list of tensors (or numpy if preferred)
        batch["embedding"] = [e.cpu() for e in embeddings]
        return batch

    # Apply vectorized map
    dataset = dataset.map(
        generate_embeddings_vectorized,
        batched=True,
        batch_size=batch_size  # adjust based on GPU memory
    )

    return dataset


def visualize_embeddings(dataset: list[tuple[str, list[float]]], n_neighbors=15, min_dist=0.1, metric="euclidean") :
    data, speaker_ids = list(zip(*dataset))    
    X = np.array(data)

    unique_speakers = sorted(set(speaker_ids))
    speaker_to_int = {spk: i for i, spk in enumerate(unique_speakers)}
    labels = np.array([speaker_to_int[s] for s in speaker_ids])

    print("transform data")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42
    )
    embedding = reducer.fit_transform(X)

    print("plot")
    plt.figure(figsize=(8, 6))

    # Color by speaker
    scatter = plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=labels,
        s=20,
        cmap="tab10"   # contains 10 qualitative colors, enough for 8 speakers
    )

    # Legend
    handles = []
    for spk in unique_speakers:
        handles.append( plt.Line2D([0], [0], marker="o", color="w", label=spk, markerfacecolor=plt.cm.tab10(speaker_to_int[spk]/10), markersize=8) )
    plt.legend(title="Speaker ID", handles=handles)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title("UMAP Projection of Vector Data by Speaker")
    plt.tight_layout()
    plt.show()
