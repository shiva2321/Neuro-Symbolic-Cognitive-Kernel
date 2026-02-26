"""
VideoAdapter — encodes video frame sequences as PerceptPackets.

Treats video as a temporal sequence of images. Each frame is encoded with
the ImageAdapter; temporal context is accumulated via TemporalStreamEncoder.
"""
from __future__ import annotations
from typing import Any, List, Optional, Sequence
import numpy as np

import python.core.vsa.hypervec_shim as hv_mod
from python.core.types.percept_packet import PerceptPacket
from python.core.types.modality_adapter import ModalityAdapter
from python.core.adapters.image_adapter import ImageAdapter, _features_to_hv, _extract_image_features


class TemporalStreamEncoder:
    """
    Encodes a stream of HVs with temporal context using exponential decay bundling.

    Each new HV is bundled with a decayed version of the accumulated state,
    creating a recency-weighted summary of the temporal stream.
    """

    def __init__(self, decay: float = 0.85) -> None:
        """
        Parameters
        ----------
        decay : float
            How much of the previous state to retain when accumulating.
            Higher = longer memory.
        """
        self.decay = decay
        self._state: Optional[hv_mod.HyperVector] = None
        self._step: int = 0

    def step(self, hv: hv_mod.HyperVector) -> hv_mod.HyperVector:
        """
        Update the temporal state with a new HV.
        Returns the new accumulated state.
        """
        if self._state is None:
            self._state = hv
        else:
            # Decay: permute the old state to shift it "back in time"
            # then bundle with the new HV
            self._state = self._state.bundle(hv)
        self._step += 1
        return self._state

    def reset(self) -> None:
        """Reset temporal state."""
        self._state = None
        self._step = 0

    @property
    def current_state(self) -> Optional[hv_mod.HyperVector]:
        return self._state

    @property
    def steps(self) -> int:
        return self._step


class VideoAdapter(ModalityAdapter):
    """
    Encodes a sequence of image frames as a single PerceptPacket
    using temporal accumulation.

    Input: list/array of frames (each frame is a 2D/3D numpy array)
    Output: PerceptPacket with modality="video"
    """

    def __init__(self, decay: float = 0.85, max_frames: int = 64) -> None:
        self._image_adapter = ImageAdapter()
        self._decay = decay
        self.max_frames = max_frames

    def encode(self, frames: Any, task_tag: str) -> PerceptPacket:
        """
        Encode a list of image frames into a single PerceptPacket.

        Parameters
        ----------
        frames : sequence of np.ndarray
            List/array of video frames.
        task_tag : str
            Task domain identifier.
        """
        if not hasattr(frames, '__iter__'):
            frames = [frames]
        frames = list(frames)[:self.max_frames]

        if len(frames) == 0:
            return PerceptPacket.make(
                modality="video",
                situation_hv=hv_mod.HyperVector(0),
                active_predicates=frozenset(["VIDEO_EMPTY"]),
                raw_state={"n_frames": 0},
            )

        encoder = TemporalStreamEncoder(decay=self._decay)
        all_preds = set()
        frame_features = []

        for frame in frames:
            frame_arr = np.asarray(frame)
            if frame_arr.ndim < 2:
                continue
            feats = _extract_image_features(frame_arr)
            frame_features.append(feats)
            frame_hv = _features_to_hv(feats)
            encoder.step(frame_hv)
            pkt = self._image_adapter.encode(frame_arr, task_tag)
            all_preds.update(pkt.active_predicates)

        temporal_hv = encoder.current_state
        if temporal_hv is None:
            temporal_hv = hv_mod.HyperVector(0)

        # Compute temporal statistics
        if frame_features:
            feat_matrix = np.stack(frame_features)
            temporal_std = feat_matrix.std(axis=0)
            motion_score = float(temporal_std.mean())
        else:
            motion_score = 0.0

        # Add video-specific predicates
        if motion_score > 0.1:
            all_preds.add("VIDEO_HIGH_MOTION")
        else:
            all_preds.add("VIDEO_LOW_MOTION")
        all_preds.add(f"VIDEO_{len(frames)}_FRAMES")

        return PerceptPacket.make(
            modality="video",
            situation_hv=temporal_hv,
            active_predicates=frozenset(all_preds),
            raw_state={
                "n_frames": len(frames),
                "motion_score": motion_score,
                "temporal_steps": encoder.steps,
            },
            adapter_name="VideoAdapter",
            adapter_trace={
                "decay": self._decay,
                "n_frames_processed": encoder.steps,
                "motion_score": motion_score,
            },
        )
