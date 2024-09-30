"""Adapted from: https://github.com/v-iashin/SpecVQGAN"""

from pathlib import Path
import typing as tp
import pickle
from math import ceil

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from torchvision.models.inception import BasicConv2d, Inception3
from torch.utils.data import DataLoader

from eval_utils.dataset import AudioDataset
from eval_utils.transforms import TRANSFORMS as ToMelSpec
from metrics.sync import repeat_audio


class Melception(Inception3):

    def __init__(
        self, num_classes, features_list, feature_extractor_weights_path, **kwargs
    ):
        # inception = Melception(num_classes=309)
        super().__init__(num_classes=num_classes, init_weights=True, **kwargs)
        self.features_list = list(features_list)
        # the same as https://github.com/pytorch/vision/blob/5339e63148/torchvision/models/inception.py#L95
        # but for 1-channel input instead of RGB.
        self.Conv2d_1a_3x3 = BasicConv2d(1, 32, kernel_size=3, stride=2)
        # also the 'hight' of the mel spec is 80 (vs 299 in RGB) we remove all max pool from Inception
        self.maxpool1 = torch.nn.Identity()
        self.maxpool2 = torch.nn.Identity()

        state_dict = torch.load(feature_extractor_weights_path, map_location="cpu")
        self.load_state_dict(state_dict["model"])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        features = {}
        remaining_features = self.features_list.copy()

        # B x 1 x 80 x 848 <- N x M x T
        x = x.unsqueeze(1)
        # (B, 32, 39, 423) <-
        x = self.Conv2d_1a_3x3(x)
        # (B, 32, 37, 421) <-
        x = self.Conv2d_2a_3x3(x)
        # (B, 64, 37, 421) <-
        x = self.Conv2d_2b_3x3(x)
        # (B, 64, 37, 421) <-
        x = self.maxpool1(x)

        if "64" in remaining_features:
            features["64"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("64")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        # (B, 80, 37, 421) <-
        x = self.Conv2d_3b_1x1(x)
        # (B, 192, 35, 419) <-
        x = self.Conv2d_4a_3x3(x)
        # (B, 192, 35, 419) <-
        x = self.maxpool2(x)

        if "192" in remaining_features:
            features["192"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("192")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        # (B, 256, 35, 419) <-
        x = self.Mixed_5b(x)
        # (B, 288, 35, 419) <-
        x = self.Mixed_5c(x)
        # (B, 288, 35, 419) <-
        x = self.Mixed_5d(x)
        # (B, 288, 35, 419) <-
        x = self.Mixed_6a(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6b(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6c(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6d(x)
        # (B, 768, 17, 209) <-
        x = self.Mixed_6e(x)

        if "768" in remaining_features:
            features["768"] = F.adaptive_avg_pool2d(x, output_size=(1, 1))
            remaining_features.remove("768")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        # (B, 1280, 8, 104) <-
        x = self.Mixed_7a(x)
        # (B, 2048, 8, 104) <-
        x = self.Mixed_7b(x)
        # (B, 2048, 8, 104) <-
        x = self.Mixed_7c(x)
        # (B, 2048, 1, 1) <-
        x = self.avgpool(x)
        # (B, 2048, 1, 1) <-
        x = self.dropout(x)

        # (B, 2048) <-
        x = torch.flatten(x, 1)

        if "2048" in remaining_features:
            features["2048"] = x
            remaining_features.remove("2048")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

        if "logits_unbiased" in remaining_features:
            # (B, num_classes) <-
            x = x.mm(self.fc.weight.T)
            features["logits_unbiased"] = x
            remaining_features.remove("logits_unbiased")
            if len(remaining_features) == 0:
                return tuple(features[a] for a in self.features_list)

            x = x + self.fc.bias.unsqueeze(0)
        else:
            x = self.fc(x)

        features["logits"] = x
        return tuple(features[a] for a in self.features_list)

    def convert_features_tuple_to_dict(self, features):
        """
        The only compound return type of the forward function amenable to JIT tracing is tuple.
        This function simply helps to recover the mapping.
        """
        message = "Features must be the output of forward function"
        assert type(features) is tuple and len(features) == len(
            self.features_list
        ), message
        return dict(
            ((name, feature) for name, feature in zip(self.features_list, features))
        )


def fid(featuresdict_1, featuresdict_2, feat_layer_name):
    eps = 1e-6

    features_1 = featuresdict_1[feat_layer_name]
    features_2 = featuresdict_2[feat_layer_name]

    assert torch.is_tensor(features_1) and features_1.dim() == 2
    assert torch.is_tensor(features_2) and features_2.dim() == 2

    stat_1 = {
        "mu": np.mean(features_1.numpy(), axis=0),
        "sigma": np.cov(features_1.numpy(), rowvar=False),
    }
    stat_2 = {
        "mu": np.mean(features_2.numpy(), axis=0),
        "sigma": np.cov(features_2.numpy(), rowvar=False),
    }

    mu1, sigma1 = stat_1["mu"], stat_1["sigma"]
    mu2, sigma2 = stat_2["mu"], stat_2["sigma"]
    assert mu1.shape == mu2.shape and mu1.dtype == mu2.dtype
    assert sigma1.shape == sigma2.shape and sigma1.dtype == sigma2.dtype

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        print(
            f"WARNING: fid calculation produces singular product; adding {eps} to diagonal of cov"
        )
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            assert False, "Imaginary component {}".format(m)
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    return {
        "frechet_inception_distance": float(fid),
    }


def get_features_dict(
    model: Melception, dataloader: DataLoader, afps: int, device: str
):
    DURATION = 9.8
    # means, stds = np.loadtxt("./data/metadata/means_stds_melspec.txt").T
    # means, stds = torch.Tensor(means)[None, :, None], torch.Tensor(stds)[None, :, None]
    out = None
    out_meta = None

    for batch in dataloader:
        metadict = {"filename": batch["filename"]}
        sample_audio = batch["sample_audio"][:, 0, ...].detach()
        sample_audio = repeat_audio(sample_audio, afps, DURATION)
        sample_spec = torch.Tensor(ToMelSpec(sample_audio.numpy()))
        sample_spec = sample_spec.permute(1, 0, 2)
        # sample_spec = (sample_spec - means) / stds
        sample_spec = sample_spec.to(device)

        with torch.no_grad():
            features = model(sample_spec)

        featuresdict = model.convert_features_tuple_to_dict(features)
        featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}

        if out is None:
            out = featuresdict
        else:
            out = {k: out[k] + featuresdict[k] for k in out.keys()}

        if out_meta is None:
            out_meta = metadict
        else:
            out_meta = {k: out_meta[k] + metadict[k] for k in out_meta.keys()}

    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    out = {**out, **out_meta}
    return out


def calculate_fid(
    sample_feats_path: tp.Optional[tp.Union[str, Path]] = None,
    gt_feats_path: tp.Optional[tp.Union[str, Path]] = None,
    sample_audio_path: tp.Optional[tp.Union[str, Path]] = None,
    gt_audio_path: tp.Optional[tp.Union[str, Path]] = None,
    duration: tp.Optional[float] = 2.56,
    metadata: tp.Optional[tp.Dict[str, float]] = None,
    apply_metadata_to_gt: bool = True,
    batch_size: int = 32,
    num_workers: int = 4,
    ckpt_path: str = "./checkpoints/melception_models/melception-21-05-10T09-28-40.pt",
    device: str = "cuda:0",
    verbose: bool = False,
    afps: int = 22_050,
) -> float:
    assert Path(ckpt_path).exists(), f"Checkpoint {ckpt_path} does not exist"
    calculate_sample_feats, calculate_gt_feats = True, True
    if sample_feats_path is None and sample_audio_path is None:
        raise ValueError(
            "Either sample_feats_path or sample_audio_path must be provided"
        )
    if gt_feats_path is None and gt_audio_path is None:
        raise ValueError("Either gt_feats_path or gt_audio_path must be provided")

    if sample_feats_path is not None and Path(sample_feats_path).exists():
        sample_feats_dict = pickle.load(open(sample_feats_path, "rb"))
        calculate_sample_feats = False

    if gt_feats_path is not None and Path(gt_feats_path).exists():
        gt_feats_dict = pickle.load(open(gt_feats_path, "rb"))
        calculate_gt_feats = False

    if calculate_sample_feats or calculate_gt_feats:
        model = Melception(
            num_classes=309,
            features_list=["2048"],
            feature_extractor_weights_path=ckpt_path,
        )
        model.to(device)

    if calculate_sample_feats:
        dataset = AudioDataset(
            audio_samples_dir=Path(sample_audio_path),
            duration=duration,
            metadata=metadata,
            apply_metadata_to_samples=False,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        sample_feats_dict = get_features_dict(model, dataloader, afps, device)
        if sample_feats_path is not None:
            pickle.dump(sample_feats_dict, open(sample_feats_path, "wb"))

    if calculate_gt_feats:
        dataset = AudioDataset(
            audio_samples_dir=Path(gt_audio_path),
            duration=duration,
            metadata=metadata,
            apply_metadata_to_samples=apply_metadata_to_gt,
        )
        dataloader = DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )
        gt_feats_dict = get_features_dict(model, dataloader, afps, device)
        if gt_feats_path is not None:
            pickle.dump(gt_feats_dict, open(gt_feats_path, "wb"))

    fid_dict = fid(sample_feats_dict, gt_feats_dict, "2048")
    if verbose:
        print("FID:", fid_dict["frechet_inception_distance"])
    return fid_dict["frechet_inception_distance"]
