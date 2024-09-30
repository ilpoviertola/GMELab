import os
from typing import Optional
from pathlib import Path
from multiprocessing.pool import ThreadPool

import torch
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
from frechet_audio_distance.utils import load_audio_task
from transformers import AutoProcessor, ASTModel
from tqdm import tqdm


def create_fad_audio_dir(fad_audio_dir: str) -> str:
    audios_path = Path(fad_audio_dir) / "fad_audios"
    audios_path.mkdir(exist_ok=False, parents=False)
    # create symbolic links to the files
    for audio in Path(fad_audio_dir).glob("*.wav"):
        (audios_path / audio.name).symlink_to(audio)
    return audios_path.as_posix()


def remove_fad_audio_dir(fad_audio_dir: str) -> None:
    # remove contents
    for audio in Path(fad_audio_dir).glob("*.wav"):
        audio.unlink()
    # remove empty dir
    Path(fad_audio_dir).rmdir()


def load_audio_files(dir, dtype="float32", sample_rate: int = 16000, channels: int = 1):
    """Adapted from Frechet Audio Distance library."""
    task_results = []

    pool = ThreadPool(10)
    pbar = tqdm(total=len(os.listdir(dir)), disable=False, desc="Loading audio files")

    def update(*a):
        pbar.update()

    print("[Frechet Audio Distance] Loading audio from {}...".format(dir))
    for fname in os.listdir(dir):
        res = pool.apply_async(
            load_audio_task,
            args=(os.path.join(dir, fname), sample_rate, channels, dtype),
            callback=update,
        )
        task_results.append(res)
    pool.close()
    pool.join()

    return [k.get() for k in task_results]


def save_ast_embeddings(
    sample_rate: int,
    audio_directory: str,
    embeddings_dir: str,
    embeddings_fn: str,
    device: str,
    use_pooler_output: bool = True,
    use_hidden_states: bool = False,
) -> None:
    assert (
        use_pooler_output + use_hidden_states == 1
    ), "Exactly one should be True from use_pooler_output and use_hidden_states"
    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    model.to(device)
    model.eval()

    audios = load_audio_files(
        audio_directory, sample_rate=sample_rate
    )  # audios are resampled here to the provided sample_rate

    embd_lst = []
    for audio in tqdm(audios, desc="Extracting AST embeddings"):
        input_values = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_values = {k: v.to(device) for k, v in input_values.items()}
        with torch.no_grad():
            output = model(**input_values)
        embd = output.pooler_output if use_pooler_output else output.last_hidden_state
        if embd.device != torch.device("cpu"):
            embd = embd.cpu()
        if torch.is_tensor(embd):
            embd = embd.squeeze(0)
            embd = embd.detach().numpy()
        embd_lst.append(embd)

    embeddings_pt = np.concatenate(embd_lst, axis=0)
    embeddings_save_path = (Path(embeddings_dir) / embeddings_fn).resolve().as_posix()
    os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
    np.save(embeddings_save_path, embeddings_pt)


def calculate_fad(
    sample_embds_path: str,
    gt_embds_path: str,
    embeddings_fn: str,
    gt_audios: Optional[str] = None,
    sample_audios: Optional[str] = None,
    model_name: str = "vggish",
    sample_rate: int = 16000,
    use_pca: bool = False,
    use_activation: bool = False,
    verbose: bool = False,
    dtype: str = "float32",
    device: str = "cuda:0",
) -> float:
    """Calculate the Frechet Audio Distance."""
    try:
        # Copy audio files (if provided) to own directories since FAD just reads the whole dir
        if gt_audios is not None:
            gt_audios = create_fad_audio_dir(gt_audios)
        if sample_audios is not None:
            sample_audios = create_fad_audio_dir(sample_audios)

        # hack to make the FAD work with AST embeddings
        if model_name.lower() == "ast":
            assert sample_rate == 16000, "AST embeddings require 16kHz sample rate"
            # initalise FAD with VGGish model (even though we are using AST embeddings)
            fad = FrechetAudioDistance(
                model_name="vggish",
                sample_rate=sample_rate,
                use_pca=False,
                use_activation=False,
                verbose=False,  # verbose,
            )
            # get AST embeddings if not already present
            if not Path(gt_embds_path + "/" + embeddings_fn).exists():
                assert (
                    gt_audios is not None
                ), "Ground truth audio directory not provided"
                save_ast_embeddings(
                    sample_rate, gt_audios, gt_embds_path, embeddings_fn, device
                )
                assert Path(
                    gt_embds_path + "/" + embeddings_fn
                ).exists(), "AST embeddings not found"
            if not Path(sample_embds_path + "/" + embeddings_fn).exists():
                assert sample_audios is not None, "Sample audio directory not provided"
                save_ast_embeddings(
                    sample_rate, sample_audios, sample_embds_path, embeddings_fn, device
                )
                assert Path(
                    sample_embds_path + "/" + embeddings_fn
                ).exists(), "AST embeddings not found"
        else:
            fad = FrechetAudioDistance(
                model_name=model_name,
                sample_rate=sample_rate,
                use_pca=use_pca,
                use_activation=use_activation,
                verbose=False,  # verbose,
            )

        score = fad.score(
            background_dir=gt_audios,
            eval_dir=sample_audios,
            background_embds_path=gt_embds_path + "/" + embeddings_fn,
            eval_embds_path=sample_embds_path + "/" + embeddings_fn,
            dtype=dtype,
        )
    except Exception as e:
        print(e)
        score = -1
    finally:
        if gt_audios is not None:
            remove_fad_audio_dir(gt_audios)
        if sample_audios is not None:
            remove_fad_audio_dir(sample_audios)

        if verbose:
            print("FAD:", score)

        assert score != -1, "FAD calculation failed."
        return score
