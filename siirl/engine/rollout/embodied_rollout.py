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

import contextlib
from contextlib import contextmanager
import os
import time
import multiprocessing
from collections import defaultdict
from datetime import datetime

import numpy as np
from siirl.params import ActorRolloutRefArguments
import tensorflow as tf
import torch
import torch.distributed
from PIL import Image
from loguru import logger
from tensordict import TensorDict
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoProcessor, GenerationConfig

from siirl.utils.model_utils.torch_functional import get_eos_mask
import siirl.utils.model_utils.torch_functional as siirl_F
from siirl.engine.rollout.base import BaseRollout
from siirl.utils.embodied.video_emb import VideoEmbeddingModel


if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from siirl.environment.embodied import LIBEROAdapter


__all__ = ['RobHFRollout']

OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@contextmanager
def _timer(name: str, timing_dict: dict):
    """A context manager to measure execution time of a code block."""
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    timing_dict[name] = timing_dict.get(name, 0) + end_time - start_time


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image):
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor and record original data type (should be tf.uint8)
    image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype

    # Convert to data type tf.float32 and values between [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Crop and then resize back to original size
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert back to PIL Image
    image = Image.fromarray(image.numpy())
    image = image.convert("RGB")
    return image


class EmbodiedHFRollout(BaseRollout):

    def __init__(self, module: nn.Module, config: ActorRolloutRefArguments):
        super().__init__()
        self.config = config
        self.model = module
        self.max_steps = {
            "libero_spatial": 512,
            "libero_object": 512,
            "libero_goal": 512,
            "libero_10": 512,
            "libero_90": 512
        }
        self._embodied_processed = False

        self._rank = torch.distributed.get_rank(
        ) if torch.distributed.is_initialized() else 0
        self._num_gpus_per_node = self.config.embodied.n_gpus_per_node

        self.embedding_model = VideoEmbeddingModel(
            model_path=config.embodied.video_embedding_model_path,
            img_size=config.embodied.embedding_img_size,
            enable_fp16=config.embodied.embedding_enable_fp16
        )

        self.enable_perf = os.environ.get("SIIRL_ENABLE_PERF", "0") == "1"
        self.embedding_model_offload = config.embodied.embedding_model_offload

        # Initialize LIBEROAdapter
        self.num_workers = self.config.embodied.env.num_envs
        # Distribute workers across available GPUs based on rank
        self.adapter = LIBEROAdapter(
            env_name=self.config.embodied.env.env_name,
            num_envs=self.num_workers,
            max_steps=self.config.embodied.env.max_steps,
            num_steps_wait=self.config.embodied.env.num_steps_wait,
            model_family=self.config.embodied.env.model_family,
            gpu_ids=[self._rank % self._num_gpus_per_node] # Run all workers on the same assigned GPU
        )
        logger.info(
            f"Initializing LIBEROAdapter with {self.num_workers} environments...")

    def close(self):
        """Gracefully shuts down the environment adapter."""
        logger.info("Closing LIBEROAdapter...")
        if hasattr(self, 'adapter') and self.adapter:
            self.adapter.close()
        logger.info("LIBEROAdapter closed.")

    def __del__(self):
        # Ensure workers are closed when the object is garbage collected
        self.close()

    def embodied_preprocess(self):
        self.processor = AutoProcessor.from_pretrained(self.config.model.path, trust_remote_code=True)

        if self.config.embodied.embodied_type in ["openvla", "openvla-oft"]:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

        if self.config.embodied.embodied_type in ["openvla-oft"]:
            if self.config.embodied.unnorm_key not in self.model.norm_stats and f"{self.config.embodied.unnorm_key}_no_noops" in self.model.norm_stats:
                self.config.embodied.unnorm_key = f"{self.config.embodied.unnorm_key}_no_noops"
            assert self.config.embodied.unnorm_key in self.model.norm_stats, f"Action un-norm key {self.config.embodied.unnorm_key} not found in VLA `norm_stats`!"

    def generate_sequences(self, prompts):
        """
        Main entry point for generating sequences.
        It splits a large batch of prompts into chunks that fit the number of workers,
        processes each chunk to generate a rollout, and then concatenates the results.
        This mimics the behavior of the original script to ensure data format compatibility.
        """
        # Preprocess the VLA model only once
        if not self._embodied_processed:
            self.embodied_preprocess()
            self._embodied_processed = True

        tic = time.time()

        total_batch_size = prompts.batch_size[0]
        n_samples = prompts['n_samples'] if 'n_samples' in prompts else 1
        assert self.num_workers >= n_samples, f"rollout num_workers({self.num_workers}) must be >= n_samples({n_samples})"
        batch_size_per_chunk = self.num_workers
        num_chunks = (total_batch_size + batch_size_per_chunk - 1) // batch_size_per_chunk
        logger.info(f"RobHFRollout.generate_sequences called with total batch size {total_batch_size}, "
                    f"n_samples {n_samples}, num_workers {self.num_workers}, batch_size_per_chunk {batch_size_per_chunk}, "
                    f"num_chunks {num_chunks}")
        
        all_chunk_outputs = []

        for i in range(num_chunks):
            start_idx = i * batch_size_per_chunk
            end_idx = min((i + 1) * batch_size_per_chunk, total_batch_size)
            
            # Slice the prompts to create a chunk
            chunk_prompts = prompts[start_idx:end_idx]
            
            logger.info(
                f"--- Processing chunk {i+1}/{num_chunks}, size = {chunk_prompts.batch_size[0]} ---")
            
            # Process one chunk and get its TensorDict output
            chunk_output = self._generate_chunk_rollout(chunk_prompts)
            all_chunk_outputs.append(chunk_output)

        # Concatenate the TensorDict objects from all chunks
        final_output = torch.cat(all_chunk_outputs)
        logger.info(f"RobHFRollout.generate_sequences finished for a single batch of size {final_output.batch_size[0]}"
                    f", took {time.time() - tic:.2f} seconds")
        return final_output

    def process_input(self,inputs:list, task_descriptions:list):
        
        batchdata = {"input_ids":[],"attention_mask":[],"pixel_values":[]}  
        
        for i in range(len(inputs)):
            input = inputs[i]
            task_description = task_descriptions[i]
           
            image = Image.fromarray(input["full_image"]).convert("RGB")
            if self.config.embodied.center_crop:
                image = center_crop_image(image)
            prompt = f"In: What action should the robot take to {task_description.lower()}?\nOut:"
            batch_feature  = self.processor(prompt, image)
            
            if "wrist_image" in input.keys():
                wrist_image = Image.fromarray(input["wrist_image"]).convert("RGB")
                if self.config.embodied.center_crop:
                    wrist_image = center_crop_image(wrist_image)
                wrist_batch_feature = self.processor(prompt, wrist_image)
                primary_pixel_values = batch_feature["pixel_values"]
                batch_feature["pixel_values"] = torch.cat([primary_pixel_values] + [wrist_batch_feature["pixel_values"]], dim=1)
                
            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]
            
            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
                )
                if self.config.embodied.embodied_type in ["openvla-oft"]:
                    attention_mask = torch.cat(
                        (attention_mask, torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(attention_mask.device)), dim=1
                    )
            
            batchdata["input_ids"].append(input_ids)    
            batchdata["attention_mask"].append(attention_mask)    
            batchdata["pixel_values"].append(pixel_values)    
        
        
        device = torch.device('cuda') 
        
        if self.config.embodied.embodied_type in ["openvla-oft"]:
            batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
            batchdata["attention_mask"] = [x.transpose(0, 1) for x in batchdata["attention_mask"]]
            batchdata["input_ids"] = pad_sequence(batchdata["input_ids"], batch_first=True, padding_value=self.processor.tokenizer.pad_token_id).squeeze(-1).to(device)
            batchdata["attention_mask"] = pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0).squeeze(-1).to(device)
            
            padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
            assert  torch.all(padding_mask==batchdata["attention_mask"].ne(0))
            padding_mask = ~padding_mask
            padding_mask = padding_mask.int() 
            sorted_indices = torch.argsort(padding_mask, dim=1, descending=True, stable=True)
            batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
            batchdata["attention_mask"] = torch.gather(batchdata["attention_mask"], 1, sorted_indices)
            
            
            batchdata["pixel_values"] = torch.cat(batchdata["pixel_values"] , dim=0).to(device)
            assert torch.all(batchdata["attention_mask"].ne(0) == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id))
        else:
            for key in ["input_ids", "attention_mask", "pixel_values"]:
                batchdata[key] = torch.cat(batchdata[key], dim=0).to(device)

        return batchdata

    def _generate_chunk_rollout(self, prompts):
        generate_tic = time.time()
        self.model.eval()
        # n_samples = prompts.get('n_samples', 1)
        task_id = prompts['task_id']
        trial_id = prompts['trial_id']
        task_suite_name = prompts['task_suite_name']
        assert np.all(task_suite_name == self.config.embodied.env.env_name), \
            "All task_suite_name in the batch must match the rollout config"
        max_steps = self.config.embodied.env.max_steps
        chunk_size = task_id.size(0)

        is_valid = "n_samples" in prompts
        global_steps = prompts.get('global_steps', 0) if is_valid else 0

        timing_dict = {}

        # Reset environments using the adapter
        with _timer(f"adapter_reset", timing_dict):
            # This is a blocking call
            init_data_list = self.adapter._blocking_reset(
                task_ids=task_id.reshape(-1).cpu().numpy().tolist(),
                trial_ids=trial_id.reshape(-1).cpu().numpy().tolist(),
            )

        inputs = [None] * chunk_size
        task_descriptions = [None] * chunk_size
        task_records = [None] * chunk_size
        valid_video = defaultdict(list)
        all_video = defaultdict(list)

        # Collect initial observations for the chunk
        with _timer(f"process_initial_obs", timing_dict):
            for idx in range(chunk_size):
                init_data = init_data_list[idx]
                task_descriptions[idx] = init_data["task_description"]
                inputs[idx] = self._obs_to_input(init_data['obs'])
                task_records[idx] = {
                    "active": init_data['active'],
                    "complete": init_data['complete'],
                    "finish_step": init_data['finish_step'],
                    "task_file_name": init_data['task_file_name']
                }
                if is_valid:
                    valid_video[init_data['task_file_name']].extend(
                        init_data['valid_images'])
                all_video[init_data['task_file_name']].extend(
                    init_data['valid_images'])

        step = 0
        vla_history = []
        meta_info_keys = ["eos_token_id", "pad_token_id", "recompute_log_prob", "validate", "do_sample", "global_steps"]
        meta_info_keys = [key for key in meta_info_keys if key in prompts.keys()]
        meta_info = prompts.select(*meta_info_keys)
        while step < max_steps:
            active_indices = [i for i, r in enumerate(task_records) if r['active']]
            current_inputs = inputs
            current_task_descriptions = task_descriptions

            with _timer(f"process_input", timing_dict):
                vla_input = self.process_input(current_inputs, current_task_descriptions)
            vla_input.update(meta_info)

            with _timer(f"_generate_one_step", timing_dict):
                vla_output = self._generate_one_step(vla_input)
            
            actions = vla_output["action"]

            step_data = {
                "responses": vla_output["responses"],
                "input_ids": vla_output["input_ids"],
                "attention_mask": vla_output["attention_mask"],
                "pixel_values": vla_output["pixel_values"],
                "action": actions,
                "step": step
            }
            vla_history.append(step_data)

            with _timer(f"adapter_step", timing_dict):
                step_results_list = self.adapter._blocking_step({
                    "indices": active_indices,
                    "actions": actions,
                })
            
            with _timer(f"process_step_results", timing_dict):
                new_inputs = inputs.copy()
                for idx in active_indices:
                    result = step_results_list[idx]
                    new_inputs[idx] = self._obs_to_input(result['obs'])
                    task_records[idx]['active'] = result['active']
                    task_records[idx]['complete'] = result['complete']
                    task_records[idx]['finish_step'] = result['finish_step']
                    all_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
                    if is_valid:
                        valid_video[task_records[idx]['task_file_name']].extend(result['valid_images'])
                inputs = new_inputs
            
            step += self.config.embodied.action_chunks_len
        
        with _timer(f"post_loop_processing", timing_dict):
            torch.cuda.empty_cache()
            self.model.train()
            
            batch = {
                    'responses': [],
                    'input_ids': [],  # here input_ids become the whole sentences
                    'attention_mask': [],
                    'pixel_values': [],
                }
            for k in ["responses", "input_ids", "attention_mask", "pixel_values"]:
                for h in vla_history:
                    batch[k].append(h[k])
            
            for k,v in batch.items():
                batch[k] = torch.stack(v,dim=1) 
    
            batch["complete"] = []
            batch["finish_step"] = []
            batch["task_file_name"] = []
    
            for k in task_records:
                batch["complete"].append(k["complete"])
                batch["finish_step"].append(k["finish_step"])
                batch["task_file_name"].append(k["task_file_name"])
            
            batch["complete"] = torch.tensor(batch["complete"], dtype=torch.bool, device=batch['responses'].device)
            batch["finish_step"] = torch.tensor(batch["finish_step"], dtype=torch.int64, device=batch['responses'].device)
            # 构建 batch
            names = batch["task_file_name"]
            max_len = 50 # max(len(n) for n in names)
            padded = [n.ljust(max_len, '\0') for n in names]
            batch["task_file_name"] = torch.tensor(
                [s.encode('utf-8') for s in padded],
                dtype=torch.uint8,
                device=batch['responses'].device
            )

        vjepa_embeddings = []
        tasks_for_embedding = []
        for k in task_records:
            tasks_for_embedding.append((
                k['task_file_name'],
                all_video.get(k['task_file_name'], []),
                "rollout4embedding",
                global_steps,
                k['complete']
            ))
        
        with _timer(f"get_embeddings", timing_dict):
            batch_names, batch_frames = zip(*[(t[0], t[1])  for t in tasks_for_embedding])
            vjepa_embeddings = self.embedding_model.get_embeddings(batch_names, batch_frames)
            batch["vjepa_embedding"] = torch.tensor(
                np.array(vjepa_embeddings), dtype=torch.float32)

        if self.enable_perf:
            generate_chunk_rollout_time = time.time() - generate_tic
            log_str = f"\n--- ⏱️  Chunk Performance (size={chunk_size}) ---\n"
            
            # Sort the dictionary by value in descending order for better readability
            sorted_timing = sorted(timing_dict.items(), key=lambda item: item[1], reverse=True)

            for key, value in sorted_timing:
                log_str += f"  {key}: {value:.4f} seconds\n"
            log_str += f"  _generate_chunk_rollout: {generate_chunk_rollout_time:.4f} seconds\n"
            log_str += f"  total steps in chunk: {step}\n"
            log_str += "--- ⏱️  End of Chunk Performance Log ---\n"

            with open(f"rollout_performance_rank_{self._rank}.log", "a") as f:
                f.write(f"\n{datetime.now()}:\n")
                f.write(log_str)
            logger.info(log_str)
        
        output_batch = TensorDict(
            batch,
            batch_size=chunk_size)
        return output_batch

    @torch.no_grad()
    def _generate_one_step(self, prompts: dict):
        if self.config.embodied.embodied_type == "openvla-oft":
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
        
        
            param_ctx = contextlib.nullcontext()

            # make sampling args can be overriden by inputs
            do_sample = prompts.get('do_sample', self.config.rollout.do_sample)
        

            temperature = prompts.get('temperature', self.config.rollout.temperature)

            #generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.model, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.model, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    actions, response = self.model.generate_action_verl(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        padding_idx = self.processor.tokenizer.pad_token_id,
                        do_sample=do_sample,
                        unnorm_key=self.config.embodied.unnorm_key,
                        temperature=temperature, )
            
            
            assert self.processor.tokenizer.pad_token_id is not None

            assert idx.ndim == 2
            idx = siirl_F.pad_sequence_to_length(idx,max_seq_len=self.config.rollout.prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            
            assert attention_mask.ndim == 2
            attention_mask = siirl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.rollout.prompt_length,pad_token_id=0,left_pad=True)
            
            
            assert idx.device.type == 'cuda'
            assert response.device.type == 'cuda'
            #assert seq.device.type == 'cuda'
            assert attention_mask.device.type == 'cuda'
            assert pixel_values.device.type == 'cuda'
            batch ={
                    'responses': response,
                    'input_ids': idx,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                }

            return batch
        
        elif self.config.embodied.embodied_type == "openvla": 
            idx = prompts['input_ids']  # (bs, prompt_length)
            attention_mask = prompts['attention_mask']  # left-padded attention_mask
            pixel_values = prompts["pixel_values"]
            
            # used to construct attention_mask
            eos_token_id = prompts['eos_token_id']
            pad_token_id = prompts['pad_token_id']

            batch_size = idx.size(0)
            prompt_length = idx.size(1)
            #self.model.eval()
            param_ctx = contextlib.nullcontext()

            do_sample = prompts.get('do_sample', self.config.rollout.do_sample)
            response_length =  self.model.get_action_dim(self.config.embodied.unnorm_key)
            top_p = prompts.get('top_p', self.config.rollout.top_p)
            top_k = prompts.get('top_k', self.config.rollout.top_k)
            if top_k is None:
                top_k = 0
            top_k = max(0, top_k)  # to be compatible with vllm

            temperature = prompts.get('temperature', self.config.rollout.temperature)
            generation_config = GenerationConfig(temperature=temperature, top_p=top_p, top_k=top_k)

            if isinstance(self.model, FSDP):
                # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
                param_ctx = FSDP.summon_full_params(self.model, writeback=False, recurse=False)
            
            with param_ctx:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    
                    output = self.model.generate(
                        input_ids=idx,
                        pixel_values=pixel_values,
                        attention_mask=attention_mask,
                        do_sample=do_sample,
                        max_new_tokens=response_length,
                        # max_length=max_length,
                        eos_token_id=eos_token_id,
                        pad_token_id=pad_token_id,
                        generation_config=generation_config,
                        # renormalize_logits=True,
                        output_scores=False,  # this is potentially very large
                        return_dict_in_generate=True,
                        use_cache=True)
                    
           
            seq = output.sequences
            sequence_length = prompt_length + response_length
            delta_length = sequence_length - seq.shape[1]
            
            assert delta_length == 0
            assert seq.shape[1] == sequence_length

            prompt = seq[:, :prompt_length]  # (bs, prompt_length)
            response = seq[:, prompt_length:]  # (bs, response_length)

            response_length = response.size(1)
            #delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
            #delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
            #response_position_ids = position_ids[:, -1:] + delta_position_id
            #position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

            response_attention_mask = get_eos_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
            attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

            # Extract predicted action tokens and translate into (normalized) continuous actions
            predicted_action_token_ids = response.detach().cpu().numpy()
            discretized_actions = self.model.vocab_size - predicted_action_token_ids
            discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.model.bin_centers.shape[0] - 1)
            normalized_actions = self.model.bin_centers[discretized_actions]

            # Unnormalize actions
            action_norm_stats = self.model.get_action_stats(self.config.embodied.unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
            action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
            actions = np.where(
                mask,
                0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
                normalized_actions,
            )
            
            actions = np.expand_dims(actions, axis=1)
            
            assert self.processor.tokenizer.pad_token_id is not None
            assert prompt.ndim == 2
            prompt = siirl_F.pad_sequence_to_length(prompt,max_seq_len=self.config.rollout.prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert seq.ndim == 2
            seq = siirl_F.pad_sequence_to_length(seq,max_seq_len=self.config.rollout.prompt_length,pad_token_id=self.processor.tokenizer.pad_token_id,left_pad=True)
            assert attention_mask.ndim == 2
            attention_mask = siirl_F.pad_sequence_to_length(attention_mask,max_seq_len=self.config.rollout.prompt_length,pad_token_id=0,left_pad=True)
            
            batch ={
                    'prompts': prompt,
                    'responses': response,
                    'input_ids': seq,
                    'attention_mask': attention_mask,
                    "pixel_values":pixel_values,
                    "action":actions,
                    #'position_ids': position_ids
                }
            
            return batch

    def _obs_to_input(self, obs):
        from siirl.utils.embodied.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle

        if self.config.embodied.num_images_in_input > 1:
            return {
                "full_image": get_libero_image(obs, 224),
                "wrist_image": get_libero_wrist_image(obs, 224),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }
        else:
            return {
                "full_image": get_libero_image(obs, 224),
                "state": np.concatenate([
                    obs["robot0_eef_pos"],
                    quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"]
                ])
            }
