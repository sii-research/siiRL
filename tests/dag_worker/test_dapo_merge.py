"""
Unit tests for DAPO filtering and merging logic.
This test simulates the real scenario from training logs to quickly verify fixes.
"""
import torch
import numpy as np
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
from siirl.data_coordinator.sample import filter_tensordict


def get_unpacked_data(value):
    """Safely get data from a value that might be NonTensorData or a raw type."""
    if isinstance(value, NonTensorData):
        return value.data
    return value


def create_mock_batch(batch_size=1024):
    """
    Create a mock batch that simulates the real data structure in DAPO.
    
    Includes:
    - Batched tensor fields (input_ids, attention_mask, etc.)
    - Batched NonTensorData fields (data_source, ability, etc.)
    - Metadata NonTensorData fields (eos_token_id, pad_token_id, etc.)
    """
    batch_dict = {
        # Tensor fields (batched)
        'input_ids': torch.randint(0, 1000, (batch_size, 10240)),
        'attention_mask': torch.ones(batch_size, 10240, dtype=torch.int64),
        'position_ids': torch.arange(10240).unsqueeze(0).repeat(batch_size, 1),
        'uid': torch.arange(batch_size, dtype=torch.int64),
        'prompts': torch.randint(0, 1000, (batch_size, 2048)),
        'responses': torch.randint(0, 1000, (batch_size, 8192)),
        'response_mask': torch.ones(batch_size, 8192, dtype=torch.int64),
        'token_level_scores': torch.randn(batch_size, 8192),
        'score': torch.randn(batch_size),
        'acc': torch.randint(0, 2, (batch_size,), dtype=torch.bool),
        'token_level_rewards': torch.randn(batch_size, 8192),
        
        # Batched NonTensorData fields (length = batch_size)
        'data_source': NonTensorData(
            data=np.array(['gsm8k'] * batch_size),
            batch_size=[batch_size]
        ),
        'ability': NonTensorData(
            data=np.array(['math'] * batch_size),
            batch_size=[batch_size]
        ),
        'reward_model': NonTensorData(
            data=np.array([{'name': 'rm1'}] * batch_size),
            batch_size=[batch_size]
        ),
        'extra_info': NonTensorData(
            data=np.array([{}] * batch_size),
            batch_size=[batch_size]
        ),
        'pred': NonTensorData(
            data=np.array(['answer'] * batch_size),
            batch_size=[batch_size]
        ),
        'global_token_num': NonTensorData(
            data=[10240] * batch_size,
            batch_size=[batch_size]
        ),
        
        # Metadata NonTensorData fields (length != batch_size)
        'eos_token_id': NonTensorData(
            data=[151643, 151645],  # Length 2, not batch_size
            batch_size=[batch_size]
        ),
        'pad_token_id': NonTensorData(
            data=151643,  # Scalar
            batch_size=[batch_size]
        ),
        'total_input_tokens': NonTensorData(
            data=168784,  # Scalar
            batch_size=[batch_size]
        ),
        'total_output_tokens': NonTensorData(
            data=966975,  # Scalar
            batch_size=[batch_size]
        ),
    }
    
    return TensorDict(batch_dict, batch_size=batch_size)


def test_filter_tensordict():
    """Test that filter_tensordict correctly handles batched data and metadata."""
    print("\n=== Testing filter_tensordict ===")
    
    # Create mock batch
    batch = create_mock_batch(batch_size=1024)
    print(f"Original batch size: {len(batch)}")
    
    # Simulate filtering (keeping ~50% of samples, like in the logs)
    indices = list(range(16, 1024, 2))  # Keep every other sample starting from 16
    print(f"Filtering to {len(indices)} samples")
    
    # Filter
    filtered_batch = filter_tensordict(batch, indices)
    print(f"Filtered batch size: {len(filtered_batch)}")
    
    # Verify tensor fields are correctly filtered
    assert len(filtered_batch['input_ids']) == len(indices)
    assert len(filtered_batch['uid']) == len(indices)
    
    # Verify batched NonTensorData fields are correctly filtered
    assert len(get_unpacked_data(filtered_batch['data_source'])) == len(indices)
    assert len(get_unpacked_data(filtered_batch['ability'])) == len(indices)
    assert len(get_unpacked_data(filtered_batch['global_token_num'])) == len(indices)
    
    # Verify metadata NonTensorData fields are preserved (not filtered)
    assert len(get_unpacked_data(filtered_batch['eos_token_id'])) == 2, "Metadata should be preserved"
    assert get_unpacked_data(filtered_batch['pad_token_id']) == 151643, "Metadata should be preserved"
    
    print("✅ filter_tensordict test passed!")
    return filtered_batch


def test_merge_cached_and_new():
    """
    Test merging cached filtered batch with new filtered batch.
    This simulates the real merge scenario in dagworker.py.
    """
    print("\n=== Testing merge logic ===")
    
    # Create first batch and filter (this will be cached)
    batch1 = create_mock_batch(batch_size=1024)
    indices1 = list(range(16, 512))  # 496 samples
    cached_batch = filter_tensordict(batch1, indices1)
    print(f"Cached batch size: {len(cached_batch)}")
    
    # Create second batch and filter (this is the new batch)
    batch2 = create_mock_batch(batch_size=1024)
    indices2 = list(range(20, 540))  # 520 samples
    new_batch = filter_tensordict(batch2, indices2)
    print(f"New batch size: {len(new_batch)}")
    
    # Now merge them (simulate dagworker merge logic)
    cache_size = len(cached_batch)
    new_size = len(new_batch)
    merged_size = cache_size + new_size
    print(f"Expected merged size: {merged_size}")
    
    merged_dict = {}
    for key in cached_batch.keys():
        cache_val = cached_batch[key]
        new_val = new_batch[key]
        
        # Debug: print info for NonTensorData fields
        if isinstance(cache_val, NonTensorData) and key in ['data_source', 'ability', 'eos_token_id']:
            print(f"\nDEBUG {key}:")
            print(f"  cache_val type: {type(cache_val)}, data type: {type(cache_val.data)}")
            if hasattr(cache_val.data, '__len__'):
                print(f"  cache_val.data length: {len(cache_val.data)}, cache_size: {cache_size}")
            print(f"  new_val type: {type(new_val)}, data type: {type(new_val.data) if isinstance(new_val, NonTensorData) else 'N/A'}")
            if isinstance(new_val, NonTensorData) and hasattr(new_val.data, '__len__'):
                print(f"  new_val.data length: {len(new_val.data)}, new_size: {new_size}")
        
        if isinstance(cache_val, torch.Tensor):
            # Merge tensors
            merged_dict[key] = torch.cat([cache_val, new_val], dim=0)
        else:
            # Handle NonTensorData or raw types (TensorDict may unpack NonTensorData)
            # Extract the actual data
            if isinstance(cache_val, NonTensorData):
                cache_data = cache_val.data
            else:
                cache_data = cache_val
            
            if isinstance(new_val, NonTensorData):
                new_data = new_val.data
            else:
                new_data = new_val
            
            # Determine if it's batched data or metadata based on length
            if isinstance(cache_data, np.ndarray):
                cache_len = len(cache_data)
                if cache_len == cache_size:
                    # Batched np.ndarray - merge
                    new_arr = new_data if isinstance(new_data, np.ndarray) else np.array(new_data)
                    merged_data = np.concatenate([cache_data, new_arr], axis=0)
                    merged_dict[key] = NonTensorData(data=merged_data, batch_size=[merged_size])
                else:
                    # Metadata - keep as NonTensorData
                    merged_dict[key] = NonTensorData(data=new_data, batch_size=[merged_size])
            elif isinstance(cache_data, (list, tuple)):
                cache_len = len(cache_data)
                if cache_len == cache_size:
                    # Batched list - merge
                    new_list = new_data if isinstance(new_data, (list, tuple)) else [new_data] * new_size
                    merged_data = list(cache_data) + list(new_list)
                    merged_dict[key] = NonTensorData(data=merged_data, batch_size=[merged_size])
                else:
                    # Metadata - keep as NonTensorData
                    merged_dict[key] = NonTensorData(data=new_data, batch_size=[merged_size])
            else:
                # Scalar - always metadata
                merged_dict[key] = NonTensorData(data=new_data, batch_size=[merged_size])
    
    # Verify all fields before creating TensorDict
    print("\n--- Pre-creation validation ---")
    for key, value in merged_dict.items():
        if isinstance(value, torch.Tensor):
            val_len = value.shape[0]
            status = "✅" if val_len == merged_size else "❌"
            print(f"{status} {key}: Tensor, shape={value.shape}")
            assert val_len == merged_size, f"Tensor {key} has wrong size: {val_len} != {merged_size}"
        elif isinstance(value, NonTensorData):
            if hasattr(value.data, '__len__') and not isinstance(value.data, str):
                val_len = len(value.data)
                # For batched data, length should match merged_size
                # For metadata, length can be different
                if val_len == merged_size:
                    print(f"✅ {key}: NonTensorData(batched), len={val_len}")
                else:
                    print(f"✅ {key}: NonTensorData(metadata), len={val_len}")
            else:
                print(f"✅ {key}: NonTensorData(scalar)")
        else:
            print(f"❌ {key}: Raw type {type(value)} - should be wrapped!")
            raise ValueError(f"Field {key} is not properly wrapped")
    
    # Try to create TensorDict
    print("\n--- Creating TensorDict ---")
    try:
        merged_batch = TensorDict(merged_dict, batch_size=merged_size)
        print(f"✅ Successfully created merged TensorDict with batch_size={merged_size}")
        
        # Verify the result
        assert len(merged_batch) == merged_size
        assert len(merged_batch['input_ids']) == merged_size
        assert len(get_unpacked_data(merged_batch['data_source'])) == merged_size
        
        # Verify metadata is preserved
        assert len(get_unpacked_data(merged_batch['eos_token_id'])) == 2
        assert get_unpacked_data(merged_batch['pad_token_id']) == 151643
        
        print("✅ Merge test passed!")
        return merged_batch
        
    except Exception as e:
        print(f"❌ Failed to create TensorDict: {e}")
        
        # Print detailed error info
        print("\n--- Error Details ---")
        for k, v in merged_dict.items():
            v_type = type(v).__name__
            v_shape = getattr(v, 'shape', 'None')
            v_dtype = getattr(v, 'dtype', 'None')
            if isinstance(v, NonTensorData):
                v_shape = f"({len(v.data)},)" if hasattr(v.data, '__len__') else 'scalar'
                v_dtype = type(v.data).__name__
            print(f"  '{k}' -> type={v_type}, shape={v_shape}, dtype={v_dtype}")
        
        raise


def test_full_dapo_workflow():
    """
    Test the complete DAPO workflow:
    1. First rollout + filter -> insufficient samples -> cache
    2. Second rollout + filter -> merge with cache -> sufficient samples
    """
    print("\n=== Testing full DAPO workflow ===")
    
    # Step 1: First rollout
    print("\n--- Step 1: First rollout ---")
    batch1 = create_mock_batch(batch_size=1024)
    indices1 = list(range(16, 512))  # 496 samples (insufficient)
    filtered1 = filter_tensordict(batch1, indices1)
    print(f"First rollout: {len(filtered1)} samples (insufficient)")
    
    # Cache it
    cached_batch = filtered1
    
    # Step 2: Second rollout
    print("\n--- Step 2: Second rollout ---")
    batch2 = create_mock_batch(batch_size=1024)
    indices2 = list(range(20, 540))  # 520 samples
    filtered2 = filter_tensordict(batch2, indices2)
    print(f"Second rollout: {len(filtered2)} samples")
    
    # Step 3: Merge
    print("\n--- Step 3: Merge cached and new ---")
    merged = test_merge_cached_and_new()
    
    print(f"\n✅ Full workflow test passed! Final batch size: {len(merged)}")


if __name__ == '__main__':
    print("=" * 60)
    print("DAPO Merge Logic Unit Tests")
    print("=" * 60)
    
    try:
        # Test 1: Filter
        filtered = test_filter_tensordict()
        
        # Test 2: Merge
        merged = test_merge_cached_and_new()
        
        # Test 3: Full workflow
        test_full_dapo_workflow()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST FAILED: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        exit(1)

