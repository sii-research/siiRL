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

from string import Template
def pre_process(tokenizer, prompt_id, obs, **kwargs):
    pre_chat_template = Template(kwargs.get("pre_chat_template", ""))
    prompt = tokenizer.decode(prompt_id)
    prompt = pre_chat_template.substitute(prompt = prompt)
    return tokenizer.encode(prompt)

def post_process(tokenizer, prompt_id, response_id, **kwargs):
    post_chat_template = kwargs.get("post_chat_template", None)
    post_chat_template_id = tokenizer.encode(post_chat_template)   
    return prompt_id + post_chat_template_id + response_id
