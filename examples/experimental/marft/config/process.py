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
