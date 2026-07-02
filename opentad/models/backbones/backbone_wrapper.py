import copy
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
import torch.utils.checkpoint as cp

from mmengine.dataset import Compose
from mmengine.registry import MODELS as MM_BACKBONES
from mmengine.runner import load_checkpoint

from ..utils import build_temporal_grid

BACKBONES = MM_BACKBONES


class BackboneWrapper(nn.Module):
    def __init__(self, cfg):
        super(BackboneWrapper, self).__init__()
        custom_cfg = cfg.custom
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")

        # build the backbone
        self.model = BACKBONES.build(model_cfg)

        # custom settings: pretrained checkpoint, post_processing_pipeline, norm_eval, freeze_backbone
        # 1. load the pretrained model
        if hasattr(custom_cfg, "pretrain") and custom_cfg.pretrain is not None:
            load_checkpoint(self.model, custom_cfg.pretrain, map_location="cpu")
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized, "
                "unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        if hasattr(custom_cfg, "pre_processing_pipeline"):
            self.pre_processing_pipeline = Compose(custom_cfg.pre_processing_pipeline)
        else:
            self.pre_processing_pipeline = None

        # 3. post_processing_pipeline for pooling and other operations
        if hasattr(custom_cfg, "post_processing_pipeline"):
            self.post_processing_pipeline = Compose(custom_cfg.post_processing_pipeline)
        else:
            self.post_processing_pipeline = None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)
        self.trainable_backbone_keywords = list(getattr(custom_cfg, "trainable_backbone_keywords", []))

        self._configure_backbone_trainability()
        self._time_debug_printed = False

        print(
            "freeze_backbone: {}, norm_eval: {}, trainable_backbone_keywords: {}".format(
                self.freeze_backbone,
                self.norm_eval,
                self.trainable_backbone_keywords,
            )
        )

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(custom_cfg, "temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_num"
            ), "temporal_checkpointing_chunk_num should be provided when using temporal checkpointing"
            assert hasattr(
                custom_cfg, "temporal_checkpointing_chunk_dim"
            ), "temporal_checkpointing_chunk_dim should be provided when using temporal checkpointing"
            self.temporal_checkpointing_chunk_num = custom_cfg.temporal_checkpointing_chunk_num
            self.temporal_checkpointing_chunk_dim = custom_cfg.temporal_checkpointing_chunk_dim

    def _configure_backbone_trainability(self):
        if not self.freeze_backbone:
            return

        keywords = tuple(self.trainable_backbone_keywords)
        for name, param in self.model.backbone.named_parameters():
            allow_grad = len(keywords) > 0 and any(keyword in name for keyword in keywords)
            param.requires_grad = allow_grad

    def _has_trainable_backbone_params(self):
        return any(param.requires_grad for param in self.model.backbone.parameters())

    def set_train_epoch(self, curr_epoch):
        if hasattr(self.model.backbone, "set_train_epoch"):
            self.model.backbone.set_train_epoch(curr_epoch)

    def _forward_backbone(self, frames, time_embed=None):
        if time_embed is None:
            return self.model.backbone(frames)
        return self.model.backbone(frames, time_embed=time_embed)

    def _build_irregular_time_features(self, frames, metas):
        if not getattr(self.model.backbone, "use_irregular_time_embed", False):
            return None
        if metas is None or len(metas) == 0:
            return None
        if not all(("irregular_selected_positions" in meta and "irregular_selected_valid_len" in meta) for meta in metas):
            return None

        processed_batches, num_segs = frames.shape[:2]
        base_batches = len(metas)
        if processed_batches % base_batches != 0:
            return None

        chunk_factor = processed_batches // base_batches
        clip_len = int(frames.shape[3])
        tubelet_size = int(getattr(self.model.backbone, "tubelet_size", 1))
        tubelet_size = max(tubelet_size, 1)
        # Match Conv3d patch embedding semantics: incomplete tail frames do not form a tubelet.
        tubelet_tokens = clip_len // tubelet_size
        if tubelet_tokens <= 0:
            return None
        effective_clip_len = tubelet_tokens * tubelet_size

        per_sample_features = []
        feat_dtype = torch.float32
        for meta in metas:
            positions = torch.as_tensor(meta.get("irregular_selected_positions", []), device=frames.device, dtype=feat_dtype).flatten()
            true_valid_len = int(positions.numel())
            dense_valid_len = int(round(float(meta.get("irregular_selected_valid_len", max(true_valid_len, 1)))))
            dense_valid_len = max(dense_valid_len, 1)
            total_frame_len = chunk_factor * clip_len

            if true_valid_len == 0:
                positions = torch.zeros(1, device=frames.device, dtype=feat_dtype)
                true_valid_len = 1

            if true_valid_len < total_frame_len:
                positions = torch.cat([positions, positions[-1:].repeat(total_frame_len - true_valid_len)], dim=0)
            else:
                positions = positions[:total_frame_len]

            frame_valid = torch.zeros(total_frame_len, device=frames.device, dtype=torch.bool)
            frame_valid[: min(true_valid_len, total_frame_len)] = True

            positions = positions.view(chunk_factor, clip_len)
            frame_valid = frame_valid.view(chunk_factor, clip_len)

            if effective_clip_len < clip_len:
                positions = positions[:, :effective_clip_len]
                frame_valid = frame_valid[:, :effective_clip_len]

            positions = positions.view(chunk_factor, tubelet_tokens, tubelet_size)
            frame_valid = frame_valid.view(chunk_factor, tubelet_tokens, tubelet_size)

            tube_valid = frame_valid.any(dim=-1)
            tube_fresh = frame_valid.all(dim=-1)
            tube_weight = frame_valid.to(feat_dtype)
            tube_center = (positions * tube_weight).sum(dim=-1) / tube_weight.sum(dim=-1).clamp_min(1.0)

            # Keep invalid tail tubelets numerically stable by copying the last valid center.
            for idx in range(1, tube_center.shape[1]):
                tube_center[:, idx] = torch.where(tube_valid[:, idx], tube_center[:, idx], tube_center[:, idx - 1])

            grid = build_temporal_grid(tube_center, valid_mask=tube_valid, fresh_mask=tube_fresh)
            point_scale = (grid["cell_left"] + grid["cell_right"]).clamp_min(1e-4)
            level_scale = grid["level_scale"].clamp_min(1e-4)[:, None]
            norm_center = tube_center / float(dense_valid_len)
            time_feat = torch.stack(
                [
                    norm_center,
                    grid["fresh_mask"].to(feat_dtype),
                    torch.log(point_scale),
                    torch.log((point_scale / level_scale).clamp_min(1e-6)),
                    torch.log((grid["cell_right"] / grid["cell_left"]).clamp_min(1e-6)),
                ],
                dim=-1,
            )
            time_feat = time_feat * grid["valid_mask"].unsqueeze(-1).to(feat_dtype)
            per_sample_features.append(time_feat[:, None].expand(chunk_factor, num_segs, tubelet_tokens, time_feat.shape[-1]))

        if len(per_sample_features) == 0:
            return None
        time_features = torch.cat(per_sample_features, dim=0).contiguous()
        if getattr(self.model.backbone, "debug_time_grid", False) and not self._time_debug_printed:
            centers = time_features[..., 0]
            fresh = time_features[..., 1] > 0.5 if time_features.shape[-1] > 1 else torch.ones_like(centers).bool()
            valid_counts = fresh.flatten(1).sum(dim=1)
            print(
                "[BackboneWrapper][TimeGrid] "
                f"shape={tuple(time_features.shape)} "
                f"valid_min={int(valid_counts.min().item()) if valid_counts.numel() > 0 else 0} "
                f"valid_max={int(valid_counts.max().item()) if valid_counts.numel() > 0 else 0} "
                f"center_min={float(centers[fresh].min().item()) if fresh.any().item() else 0.0:.4f} "
                f"center_max={float(centers[fresh].max().item()) if fresh.any().item() else 0.0:.4f}"
            )
            self._time_debug_printed = True
        return time_features

    def forward(self, frames, masks=None, metas=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # set all normalization layers
        self.set_norm_layer()

        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        time_embed = self._build_irregular_time_features(frames, metas)

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]
        if time_embed is not None:
            time_embed = time_embed.flatten(0, 1).contiguous()

        # go through the video backbone
        if self.freeze_backbone and not self._has_trainable_backbone_params():  # freeze everything even in training
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    features = self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                        time_embed=time_embed,
                    )
                else:
                    features = self._forward_backbone(frames, time_embed=time_embed)

        else:  # let the model.train() or model.eval() decide whether to freeze
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                    time_embed=time_embed,
                )
            else:
                features = self._forward_backbone(frames, time_embed=time_embed)

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat([self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1)
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def tensor_to_list(self, tensor):
        return [t for t in tensor]

    def unflatten_and_pool_features(self, features, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(dim=0, sizes=(batches, num_segs))  # [bs, num_seg, ...]

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim, time_embed=None):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and sequentially forward each chunk with
        no gradients. 2) The backward pass will recompute the intermediate activations and compute each chunk's gradient. 3) Backbone's
        gradients will be accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        if time_embed is not None and chunk_dim != 0:
            raise NotImplementedError("Backbone time embedding only supports temporal_checkpointing with chunk_dim=0.")

        def _inner_forward(frames):
            return self.model.backbone(frames)

        def _inner_forward_with_time(frames, chunk_time_embed):
            return self.model.backbone(frames, time_embed=chunk_time_embed)

        video_feat = []
        frame_chunks = torch.chunk(frames, chunk_num, dim=chunk_dim)
        if time_embed is not None:
            time_chunks = torch.chunk(time_embed, chunk_num, dim=0)
        else:
            time_chunks = [None] * len(frame_chunks)

        for mini_frames, mini_time_embed in zip(frame_chunks, time_chunks):  # B*N is chunked
            # we can use torch.cp.checkpoint to implement an efficient temporal checkpointing mechanism
            if mini_time_embed is None:
                mini_feat = cp.checkpoint(_inner_forward, mini_frames, use_reentrant=False)
            else:
                mini_feat = cp.checkpoint(_inner_forward_with_time, mini_frames, mini_time_embed, use_reentrant=False)
            video_feat.append(mini_feat)

        if isinstance(video_feat[0], (tuple, list)):
            video_feat = [torch.cat([f[idx] for f in video_feat], dim=chunk_dim) for idx in range(len(video_feat[0]))]
        else:
            video_feat = torch.cat(video_feat, dim=chunk_dim)
        return video_feat
