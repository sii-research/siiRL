# Copyright 2025, Shanghai Innovation Institute. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from loguru import logger

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.vision_transformer import vit_giant_xformers_rope

# Constants for video normalization
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class VideoEmbeddingModel:
    """
    A self-contained class to load the V-JEPA model, preprocess video frames,
    and extract embeddings. Each instance is tied to a specific GPU device.
    """
    def __init__(self, model_path: str, img_size: int = 384, device_id: int = 0, enable_fp16: bool = False) -> None:
        self.model_path = model_path
        self.img_size = img_size
        self.device = f"cuda:{device_id}"
        self.auto_cast_dtype = torch.float16 if enable_fp16 else torch.float32
        logger.info(f"Initializing embedding model on device: {self.device}, enbale_fp16={enable_fp16}. "
                    "It will take several minutes, please be patient...")
        self.pt_video_transform, self.model_pt = self._create_model_instance()
        self.embedding_dim = self.model_pt.norm.bias.shape[0]
        self.num_frames_for_embedding = 64
        logger.info(f"Embedding model loaded successfully on {self.device}")

    def _build_pt_video_transform(self):
        """Builds the video transformation pipeline for the model."""
        short_side_size = int(256.0 / 224 * self.img_size)
        eval_transform = video_transforms.Compose(
            [
                video_transforms.Resize(short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(self.img_size, self.img_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ]
        )
        return eval_transform

    def _load_pretrained_vjepa_pt_weights(self, model, pretrained_weights):
        """Loads pretrained weights into the V-JEPA model."""
        try:
            pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location=self.device)["encoder"]
            pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
            pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
            msg = model.load_state_dict(pretrained_dict, strict=False)
            logger.info(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")
        except Exception as e:
            logger.error(f"Failed to load pretrained weights from {pretrained_weights}: {e}")
            raise

    def _create_model_instance(self):
        """Creates and prepares the V-JEPA model instance."""
        model_pt = vit_giant_xformers_rope(
            img_size=(self.img_size, self.img_size),
            num_frames=64)
        self._load_pretrained_vjepa_pt_weights(model_pt, self.model_path)
        model_pt.eval().to(self.device)
        pt_video_transform = self._build_pt_video_transform()
        return pt_video_transform, model_pt

    def offload_to_host(self):
        """Offloads the model to CPU to free up GPU memory."""
        self.model_pt.to("cpu")
        torch.cuda.empty_cache()
        logger.info(f"Video embedding model offloaded to CPU.")

    def load_to_device(self):
        """Loads the model back to the assigned GPU device."""
        self.model_pt.to(self.device)
        logger.info(f"Video embedding model loaded back to {self.device}.")

    def extract_video_embedding(self, video_tensor):
        """
        Extracts the embedding from a given video tensor.
        Args:
            video_tensor (torch.Tensor): A tensor of shape (T, C, H, W).
        Returns:
            np.ndarray: The computed embedding vector.
        """
        if video_tensor is None:
            return None
        with torch.inference_mode():
            # The transform expects a tensor on the correct device
            x = self.pt_video_transform(video_tensor.to(self.device)).to(self.device).unsqueeze(0)
            with torch.amp.autocast('cuda', dtype=self.auto_cast_dtype):
                embedding = self.model_pt(x)
            return embedding.mean(dim=1).to(torch.float32).squeeze(0).cpu().numpy()

    def extract_video_embedding_batch(self, video_tensor_list):
        """
        Extracts embeddings for a batch of video tensors.
        Args:
            video_tensor_list (list of torch.Tensor or None): List of video tensors.
        Returns:
            list of np.ndarray: List of computed embedding vectors.
        """
        with torch.inference_mode():
            input_list = [self.pt_video_transform(v.to(self.device)).to(self.device) for v in video_tensor_list]
            x = torch.stack(input_list, dim=0)
            with torch.amp.autocast('cuda', dtype=self.auto_cast_dtype):
                embedding = self.model_pt(x)
            return [e.mean(dim=0).to(torch.float32).cpu().numpy() for e in embedding]
        
    def get_embeddings(self, batch_names, batch_frames):
        """
        Processes video frames in memory and returns embeddings for each task.
        Handles missing frames gracefully and batches all embedding extractions for efficiency.
        All videos are processed in a single batch after padding shorter videos.
        Args:
            batch_names (list of str): List of task names or identifiers.
            batch_frames (list of list of np.ndarray): List of video frames for each task.
        Returns:
            list of np.ndarray: List of embedding vectors for each task.
        """
        assert len(batch_names) == len(batch_frames), "Names and frames lists must be of the same length."
        embedding_list = [np.zeros((self.embedding_dim), dtype=np.float32) for _ in batch_names]
        
        video_tensors_to_process = []
        original_indices = []

        for idx, (name, frames) in enumerate(zip(batch_names, batch_frames)):
            if not frames:
                logger.warning(f"== Found 0 frames for video {name}, returning zero embedding ==")
                continue
            try:
                total_frames = len(frames)
                if total_frames >= self.num_frames_for_embedding:
                    selected_indices = np.linspace(0, total_frames - 1, num=self.num_frames_for_embedding, dtype=int)
                    sampled_frames = [frames[i] for i in selected_indices]
                else:
                    logger.warning(f"Video {name} has only {total_frames} frames. Padding to {self.num_frames_for_embedding}.")
                    indices = np.arange(total_frames)
                    padded_indices = np.resize(indices, self.num_frames_for_embedding)
                    sampled_frames = [frames[i] for i in padded_indices]
                # Convert list of numpy arrays to a single torch tensor
                # The frames are (H, W, C), convert to (T, C, H, W)
                video_tensor = torch.from_numpy(np.stack(sampled_frames)).permute(0, 3, 1, 2)
                video_tensors_to_process.append(video_tensor)
                original_indices.append(idx)
            except Exception as e:
                logger.error(f"Error processing frames for {name}: {e}")

        if not video_tensors_to_process:
            return embedding_list

        batch_embeddings = self.extract_video_embedding_batch(video_tensors_to_process)
        for i, emb in zip(original_indices, batch_embeddings):
            embedding_list[i] = emb

        return embedding_list
