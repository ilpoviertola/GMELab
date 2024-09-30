from pathlib import Path
from typing import Optional, Dict, List

from torch.nn.functional import pad
from torch.utils.data import Dataset
from torchaudio import load


class AudioWithGtDataset(Dataset):
    """PyTorch Dataset for audio files."""

    def __init__(
        self,
        audio_samples_dir: Path,
        audio_gts_dir: Path,
        duration: Optional[float] = 2.56,
        metadata: Optional[Dict] = None,
        apply_metadata_to_samples: bool = False,
    ):
        """Initialize AudioDataset.

        Args:
            audio_samples_dir (Path): Path to audio samples file dir.
            audio_gts_dir (Path): Path to GT audio files dir.
            duration (float, optional): Duration of audio files in seconds. Defaults to 2.0.
        """

        audio_samples = list(audio_samples_dir.glob("*.wav"))
        audio_gts = list(audio_gts_dir.glob("*.wav"))
        assert len(audio_samples) <= len(
            audio_gts
        ), "Must have same number of samples or less than ground truths."

        self.audio_samples: List[Path] = sorted(audio_samples, key=lambda p: p.name)
        self.audio_gts_dir = Path(audio_gts_dir)
        self.duration = duration
        self.metadata = metadata if metadata is not None else {}
        self.apply_metadata_to_samples = apply_metadata_to_samples

    def __len__(self):
        """Return length of dataset."""
        return len(self.audio_samples)

    def __getitem__(self, idx):
        """Return item at index idx."""
        sample = self.audio_samples[idx]
        gt: Path = self.audio_gts_dir / sample.name
        assert gt.exists(), "Ground truth audio file does not exist."
        start_sec = float(
            self.metadata.get(gt.stem, 0)
        )  # this is where sample starts in gt

        sample_audio, sample_audio_sr = load(sample)
        if self.apply_metadata_to_samples:
            sample_audio = sample_audio[..., int(start_sec * sample_audio_sr) :]
        gt_audio, gt_audio_sr = load(gt)
        gt_audio = gt_audio[..., int(start_sec * gt_audio_sr) :]

        sample_audio = pad(
            sample_audio,
            (0, int(self.duration * sample_audio_sr) - sample_audio.shape[-1]),
            mode="constant",
            value=0,
        )
        gt_audio = pad(
            gt_audio,
            (0, int(self.duration * gt_audio_sr) - gt_audio.shape[-1]),
            mode="constant",
            value=0,
        )
        if gt_audio.shape[0] == 2:
            gt_audio = gt_audio.mean(dim=0, keepdim=True)
        if sample_audio.shape[0] == 2:
            sample_audio = sample_audio.mean(dim=0, keepdim=True)
        return {
            "sample_audio": sample_audio,
            "gt_audio": gt_audio,
            "sample_audio_sr": sample_audio_sr,
            "gt_audio_sr": gt_audio_sr,
            "filename": sample.name,
        }


class AudioDataset(Dataset):
    """PyTorch Dataset for audio files."""

    def __init__(
        self,
        audio_samples_dir: Path,
        duration: Optional[float] = 2.56,
        metadata: Optional[Dict] = None,
        apply_metadata_to_samples: bool = False,
    ):
        """Initialize AudioDataset.

        Args:
            audio_samples_dir (Path): Path to audio samples file dir.
            duration (float, optional): Duration of audio files in seconds. Defaults to 2.0.
        """

        audio_samples = list(audio_samples_dir.glob("*.wav"))
        assert len(audio_samples) > 0, "No audio samples found."

        self.audio_samples: List[Path] = sorted(audio_samples, key=lambda p: p.name)
        self.duration = duration
        self.metadata = metadata if metadata is not None else {}
        self.apply_metadata_to_samples = apply_metadata_to_samples

    def __len__(self):
        """Return length of dataset."""
        return len(self.audio_samples)

    def __getitem__(self, idx):
        """Return item at index idx."""
        sample = self.audio_samples[idx]
        start_sec = float(
            self.metadata.get(sample.stem, 0)
        )  # this is where sample starts

        sample_audio, sample_audio_sr = load(sample)
        if self.apply_metadata_to_samples:
            sample_audio = sample_audio[..., int(start_sec * sample_audio_sr) :]

        if self.duration is not None:
            sample_audio = pad(
                sample_audio,
                (0, int(self.duration * sample_audio_sr) - sample_audio.shape[-1]),
                mode="constant",
                value=0,
            )
        if sample_audio.shape[0] == 2:
            sample_audio = sample_audio.mean(dim=0, keepdim=True)
        return {
            "sample_audio": sample_audio,
            "sample_audio_sr": sample_audio_sr,
            "filename": sample.name,
        }
