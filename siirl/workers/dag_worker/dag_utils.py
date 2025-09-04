from siirl.workers.databuffer import DataProto
from siirl.workers.dag.node import Node
from tensordict import TensorDict


def add_prefix_to_dataproto(data_proto: DataProto, node: Node):
    """
    Adds a prefix to all keys in the DataProto.
    The prefix is formatted as f"agent_group_{node.agent_group}_".
    Only keys that do not already have a prefix will be modified.

    Args:
        data_proto (DataProto): The DataProto instance.
        node (Node): The node containing the agent_group.
    """
    prefix = f"agent_group_{node.agent_group}_"
    prefix_agent_group = "agent_group_"

    # Process tensor batch
    if data_proto.batch is not None:
        new_batch = {}
        for key, value in data_proto.batch.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_batch[new_key] = value
            else:
                new_batch[key] = value
        data_proto.batch = TensorDict(new_batch, batch_size=data_proto.batch.batch_size)

    # # Process non_tensor_batch
    if data_proto.non_tensor_batch is not None:
        new_non_tensor = {}
        for key, value in data_proto.non_tensor_batch.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_non_tensor[new_key] = value
            else:
                new_non_tensor[key] = value
        data_proto.non_tensor_batch = new_non_tensor

    # # Process meta_info
    if data_proto.meta_info is not None:
        new_meta = {}
        for key, value in data_proto.meta_info.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_meta[new_key] = value
            else:
                new_meta[key] = value
        data_proto.meta_info = new_meta
    return data_proto


def remove_prefix_from_dataproto(data_proto, node: Node):
    """
    Removes the prefix from all keys in the DataProto.
    Only keys with a matching prefix will have the prefix removed.

    Args:
        data_proto (DataProto): The DataProto instance.
        node (Node): The node containing the agent_group to identify the prefix.
    """
    prefix = f"agent_group_{node.agent_group}_"
    prefix_len = len(prefix)

    # Process tensor batch
    if data_proto.batch is not None:
        new_batch = {}
        for key, value in data_proto.batch.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_batch[new_key] = value
            else:
                new_batch[key] = value
        data_proto.batch = TensorDict(new_batch, batch_size=data_proto.batch.batch_size)

    # Process non_tensor_batch
    if data_proto.non_tensor_batch is not None:
        new_non_tensor = {}
        for key, value in data_proto.non_tensor_batch.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_non_tensor[new_key] = value
            else:
                new_non_tensor[key] = value
        data_proto.non_tensor_batch = new_non_tensor

    # # Process meta_info
    if data_proto.meta_info is not None:
        new_meta = {}
        for key, value in data_proto.meta_info.items():
            if key.startswith(prefix):
                new_key = key[prefix_len:]
                new_meta[new_key] = value
            else:
                new_meta[key] = value
        data_proto.meta_info = new_meta

    return data_proto

def add_prefix_to_metrics(metrics: dict, node: Node):
    """
    Adds a prefix to all keys in the metrics.
    The prefix is formatted as f"agent_group_{node.agent_group}_".
    Only keys that do not already have a prefix will be modified.

    Args:
        metrics (Dict): The metrics instance.
        node (Node): The node containing the agent_group.
    """
    prefix = f"agent_{node.agent_group}_"
    prefix_agent_group = "agent_"
    if metrics:
        new_metrics = {}
        for key, value in metrics.items():
            if not key.startswith(prefix_agent_group):
                new_key = prefix + key
                new_metrics[new_key] = value
            else:
                new_metrics[key] = value
        metrics = new_metrics
    return metrics