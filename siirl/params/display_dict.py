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

import json
from loguru import logger
from typing import Dict, Any, Optional, List, Tuple

# --- Formatting Constants ---
BASE_INDENT_UNIT_FOR_LOGGING = "  "
DESIRED_MIN_VARIABLE_DOTS_FOR_FILLER = 1  # Minimum dots in the "..." part for simple values
DEFAULT_HEADER_TEXT_LOGGING = "details"
TARGET_HEADER_TOTAL_WIDTH_LOGGING = 80  # Adjusted for wider output
TARGET_VALUE_ALIGNMENT_COLUMN_LOGGING = 80  # Target column for value alignment for simple values, lists, and sets.


def _render_dict_recursively_util(current_dict_to_render: Dict[str, Any], current_indent_str: str, fixed_value_align_col: int, base_indent_unit: str, lines: list):
    """
    Internal recursive helper to render dictionary content.
    Dictionaries are expanded. Lines announcing a dictionary now end with a colon.
    Lists and sets are printed as single-line strings aligned to fixed_value_align_col.
    Simple values are also aligned to fixed_value_align_col.
    """
    try:
        sorted_items: List[Tuple[str, Any]] = sorted([(str(k), v) for k, v in current_dict_to_render.items()])
    except Exception as e:
        lines.append(f"{current_indent_str}[Could not sort keys: {e}]")
        sorted_items = [(str(k), v) for k, v in current_dict_to_render.items()]

    if not sorted_items and current_indent_str != base_indent_unit:
        lines.append(f"{current_indent_str}(empty dict)")
        return

    for key_s, value_obj in sorted_items:
        prefix_key_only = f"{current_indent_str}{key_s}"

        if isinstance(value_obj, dict):
            lines.append(f"{prefix_key_only}:")
            _render_dict_recursively_util(value_obj, current_indent_str + base_indent_unit, fixed_value_align_col, base_indent_unit, lines)
        else:
            if isinstance(value_obj, list):
                try:
                    val_s = json.dumps(value_obj, separators=(",", ":"), ensure_ascii=False)
                except TypeError:
                    val_s = str(value_obj)
            elif isinstance(value_obj, set):
                try:
                    val_s = json.dumps(sorted(list(value_obj)), separators=(",", ":"), ensure_ascii=False)
                except TypeError:
                    val_s = str(value_obj)
            else:
                val_s = str(value_obj)

            prefix_for_dots_alignment = f"{prefix_key_only} ."
            suffix_part_len = len(". ") + len(val_s)
            dots_needed = fixed_value_align_col - len(prefix_for_dots_alignment) - suffix_part_len
            dots = "." * max(DESIRED_MIN_VARIABLE_DOTS_FOR_FILLER, dots_needed)
            lines.append(f"{prefix_for_dots_alignment}{dots}. {val_s}")


def log_dict_formatted(data_dict: Dict[str, Any], title: Optional[str] = "Configuration Details", header_text_content: str = DEFAULT_HEADER_TEXT_LOGGING, target_value_alignment_column: int = TARGET_VALUE_ALIGNMENT_COLUMN_LOGGING, log_level: str = "info"):
    """
    Logs a dictionary with hierarchical indentation for nested dictionaries,
    styled similarly to Megatron-LM argument printing. Uses loguru.
    Lines announcing a nested dictionary end with a colon.
    Lists, sets, and simple values are aligned to target_value_alignment_column.

    Args:
        data_dict (Dict[str, Any]): The dictionary to log.
        title (Optional[str]): A title for this configuration block.
        header_text_content (str): Text to use in the header/footer lines.
        target_value_alignment_column (int): The column index where simple values/lists/sets should start.
    """
    if not isinstance(data_dict, dict):
        logger.error(f"Invalid input: data_dict must be a dictionary. Received type: {type(data_dict)}")
        return

    current_target_header_width = max(TARGET_HEADER_TOTAL_WIDTH_LOGGING, target_value_alignment_column + 10)
    num_spaces_in_header = 2
    padding_dashes_total = current_target_header_width - len(header_text_content) - num_spaces_in_header
    if padding_dashes_total < 0:
        padding_dashes_total = 0

    dashes_left = padding_dashes_total // 2
    dashes_right = padding_dashes_total - dashes_left
    header_line = f"{'-' * dashes_left} {header_text_content} {'-' * dashes_right}"

    lines = []
    if title:
        lines.append(f"## {title} ##")
    lines.append(header_line)

    if not data_dict:
        lines.append(f"{BASE_INDENT_UNIT_FOR_LOGGING}(No items in dictionary)")
    else:
        _render_dict_recursively_util(data_dict, BASE_INDENT_UNIT_FOR_LOGGING, target_value_alignment_column, BASE_INDENT_UNIT_FOR_LOGGING, lines)

    lines.append("-" * len(header_line))
    lines.append("")

    # Log all lines as a single multi-line message
    valid_levels = ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(f"Invalid log level: {log_level}. Choose from {valid_levels}.")
    logger.log(log_level.upper(), "\n".join(lines))
