import pytest
import torch
import numpy as np
from tensordict import TensorDict
from tensordict.tensorclass import NonTensorData
import logging

# Configure logging for tests
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

from siirl.data_coordinator.sample import preprocess_dataloader

def test_preprocess_dataloader():
    """
    Tests if preprocess_dataloader correctly handles data repetition (n > 1)
    and creates integer uids.
    """
    data = {
        'input_ids': np.array([[1, 2], [3, 4]]),
        'attention_mask': torch.tensor([[1, 1], [1, 0]]),
        'data_source': ['d1', 'd2'] 
    }
    n = 2
    
    tensor_dict = preprocess_dataloader(data, n=n)
    
    # 1. Check batch size
    assert tensor_dict.batch_size[0] == 4
    
    # 2. Check uid creation and type
    assert 'uid' in tensor_dict.keys()
    assert len(tensor_dict['uid']) == 4
    # uids should be [0, 0, 1, 1] after repeat
    expected_uids = np.array([0, 0, 1, 1], dtype=np.int64)
    assert np.array_equal(tensor_dict['uid'], expected_uids)

    # 3. Check numpy array repetition
    expected_input_ids = np.array([[1, 2], [1, 2], [3, 4], [3, 4]])
    assert np.array_equal(tensor_dict['input_ids'], expected_input_ids)
    
    # 4. Check torch tensor repetition
    expected_attention_mask = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 0]])
    assert torch.equal(tensor_dict['attention_mask'], expected_attention_mask)
    
    # 5. Check NonTensorData repetition
    assert isinstance(tensor_dict['data_source'], np.ndarray)
    expected_data_source = np.array(['d1', 'd1', 'd2', 'd2'])
    assert np.array_equal(tensor_dict['data_source'], expected_data_source)


# Mock DAGWorker for testing postprocess_sampling
class MockDAGWorker:
    def __init__(self, rank):
        self._rank = rank
        self.sampling_leftover_cache = None

    # Simplified version of the method for testing purposes
    def postprocess_sampling(self, config, batch: TensorDict, filtered_indices: list):
        
        # Mock dynamic_sampling behavior
        filtered_batch = batch[filtered_indices]
        metrics = {
            'dapo_sampling/kept_trajectories_ratio': len(filtered_indices) / len(batch),
            'dapo_sampling/filtered_indices': filtered_indices # Simulate returning indices in metrics
        }

        # --- This part now mirrors the production code ---
        
        # Get the indices from the metrics dict
        local_filtered_indices = metrics.pop('dapo_sampling/filtered_indices', [])

        if self.sampling_leftover_cache is not None:
            # Manual merge logic from the actual implementation
            from tensordict.tensorclass import NonTensorData
            cache_size = len(self.sampling_leftover_cache)
            new_size = len(filtered_batch)
            merged_size = cache_size + new_size
            
            merged_dict = {}
            # Use keys from the new batch as it's guaranteed to be non-empty
            for key in set(self.sampling_leftover_cache.keys()) | set(filtered_batch.keys()):
                cache_val = self.sampling_leftover_cache.get(key)
                new_val = filtered_batch.get(key)

                print(f"\n--- Processing key: {key} ---")
                
                if cache_val is None:
                    merged_dict[key] = new_val
                    continue
                if new_val is None:
                    merged_dict[key] = cache_val
                    continue

                print(f"  Cache type: {type(cache_val)}, New type: {type(new_val)}")

                # --- Final, explicit merge logic ---
                if isinstance(cache_val, torch.Tensor):
                    print("  Type is torch.Tensor. Concatenating.")
                    merged_dict[key] = torch.cat([cache_val, new_val], dim=0)
                elif isinstance(cache_val, NonTensorData):
                    # Check the type of the wrapped data
                    if isinstance(cache_val.data, np.ndarray):
                        # It's a numpy array wrapped in NonTensorData.
                        new_val_filtered = np.array(batch[key].data)[local_filtered_indices]
                        merged_data = np.concatenate([cache_val.data, new_val_filtered], axis=0)
                        merged_dict[key] = NonTensorData(data=merged_data, batch_size=torch.Size([merged_size]))
                    elif isinstance(cache_val.data, (list, tuple)):
                        # It's batched list-like data.
                        # new_val is already the filtered list
                        merged_data = cache_val.tolist() + new_val
                        merged_dict[key] = NonTensorData(data=merged_data, batch_size=torch.Size([merged_size]))
                    else:
                        # It's metadata, keep the new value
                        merged_dict[key] = new_val
                else:
                    # Fallback for any other metadata (simple int, str, etc.)
                    print(f"  Type is other metadata ({type(cache_val)}). Keeping new value.")
                    merged_dict[key] = new_val
            
            print("--- Merge Complete ---")
            filtered_batch = TensorDict(merged_dict, batch_size=torch.Size([merged_size]))
            self.sampling_leftover_cache = None
        
        # Mock distributed aggregation and decision
        total_samples = len(filtered_batch) # Simplified for single-process test
        target_total_samples = config['target_total_samples']

        if total_samples < target_total_samples:
            self.sampling_leftover_cache = filtered_batch
            return TensorDict({}, batch_size=(0,)), metrics
        else:
            return filtered_batch, metrics

@pytest.fixture
def sample_batch():
    """Provides a sample TensorDict for testing."""
    return TensorDict({
        'uid': torch.arange(8),
        'input_ids': torch.randn(8, 10),
        'data_source': NonTensorData(np.array([f"src_{i}" for i in range(8)]), batch_size=8)
    }, batch_size=8)


def test_postprocess_sampling_caching(sample_batch):
    """
    Tests if postprocess_sampling correctly caches data when there are insufficient samples.
    """
    worker = MockDAGWorker(rank=0)
    config = {'target_total_samples': 10}
    
    # First pass: not enough samples, should cache and return empty
    filtered_indices = [1, 3, 5]
    result_batch, _ = worker.postprocess_sampling(config, sample_batch, filtered_indices)
    
    assert len(result_batch) == 0
    assert worker.sampling_leftover_cache is not None
    assert len(worker.sampling_leftover_cache) == 3
    assert torch.equal(worker.sampling_leftover_cache['uid'], torch.tensor([1, 3, 5]))

def test_postprocess_sampling_merging(sample_batch):
    """
    Tests if postprocess_sampling correctly merges cached data with new data.
    """
    worker = MockDAGWorker(rank=0)
    config = {'target_total_samples': 5}
    
    # First, cache some data
    worker.sampling_leftover_cache = TensorDict({
        'uid': torch.tensor([10, 20]),
        'input_ids': torch.randn(2, 10),
        'data_source': NonTensorData(np.array(['cached_1', 'cached_2']), batch_size=2),
        'metadata_field': NonTensorData(data=1, batch_size=torch.Size([2])) # Wrap metadata in NonTensorData
    }, batch_size=2)
    
    # Add metadata to the new batch as well
    sample_batch['metadata_field'] = NonTensorData(data=2, batch_size=sample_batch.batch_size) # New metadata
    
    # Second pass: new data should be merged with cache
    filtered_indices = [0, 2, 4, 6]
    result_batch, _ = worker.postprocess_sampling(config, sample_batch, filtered_indices)
    
    # Total samples (2 cached + 4 new = 6) >= target (5), so should return merged batch
    assert worker.sampling_leftover_cache is None
    assert len(result_batch) == 6
    
    # Check merged content
    expected_uids = torch.tensor([10, 20, 0, 2, 4, 6])
    assert torch.equal(result_batch['uid'], expected_uids)
    
    expected_sources = ['cached_1', 'cached_2', 'src_0', 'src_2', 'src_4', 'src_6']
    assert np.array_equal(result_batch['data_source'].data, np.array(expected_sources))
    
    # Check that the metadata from the NEW batch is kept
    assert result_batch['metadata_field'] == 2
