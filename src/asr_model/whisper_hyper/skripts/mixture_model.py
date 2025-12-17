from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np
import src.magic_strings as magic_strings
import torch
from transformers import WhisperFeatureExtractor, WhisperModel
from numpy.typing import NDArray
from typing import Literal
from datasets import Dataset
import torch
import torchaudio
import umap
import matplotlib.pyplot as plt
import os

type CovarianceType = Literal['full', 'tied', 'diag', 'spherical']

def fit_GMM(embeddings: list[list[float]], n_components: int, covariance_type: CovarianceType='full') :

    embeddings_array = np.asarray(embeddings)

    gmm = GaussianMixture(n_components=n_components, covariance_type=covariance_type, random_state=42)
    gmm.fit(embeddings_array)

    return gmm


def fit_BGMM(embeddings: list[list[float]], max_components=100, covariance_type: CovarianceType='full', weight_concentration_prior=1.0):
    embeddings_array: NDArray = np.asarray(embeddings)

    model = BayesianGaussianMixture(
        n_components=max_components,
        covariance_type=covariance_type,
        weight_concentration_prior_type='dirichlet_process',
        weight_concentration_prior=weight_concentration_prior,   # higher values = softer predictions
        max_iter=500,
        init_params='kmeans' # can be "random"
    )

    model.fit(embeddings_array)    
    
    return model


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

import colorsys

def visualize_embeddings(
    labels: list[str],
    embeddings: list[list[float]],
    n_neighbors: int = 10,
    min_dist: float = 0.2,
    metric: str = "euclidean",
    save_dir: str | None = None,
    filename: str = None, 
    title: str="UMAP Projection",
    show: bool = True,
):
    X = np.array(embeddings)

    unique_labels = sorted(set(labels))
    n_labels = len(unique_labels)

    label_to_int = {lbl: i for i, lbl in enumerate(unique_labels)}
    label_indices = np.array([label_to_int[lbl] for lbl in labels])

    # Random color assignment (one color per label)
    hues = np.linspace(0.0, 1.0, n_labels, endpoint=False)
    colors = np.array([ colorsys.hsv_to_rgb(h, 0.75, 0.95) for h in hues ])

    print("transform data")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=42,
    )
    embedding = reducer.fit_transform(X)

    print("plot")
    plt.figure(figsize=(8, 6))

    point_colors = colors[label_indices]

    plt.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=point_colors,
        s=20,
    )

    # Legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=lbl,
            markerfacecolor=colors[label_to_int[lbl]],
            markersize=8,
        )
        for lbl in unique_labels
    ]
    plt.legend(title="Speaker ID", handles=handles)

    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.title(title)
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
