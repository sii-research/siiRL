def pre_process(tokenizer, prompt_id, **kwargs):
    pre_chat_template = kwargs.get("pre_chat_template", None)
    prompt = tokenizer.decode(prompt_id)
    prompt = pre_chat_template.format(prompt = prompt)
    return tokenizer.encode(prompt)

def post_process(tokenizer, prompt_id, response_id, **kwargs):
    post_chat_template = kwargs.get("post_chat_template", None)
    post_chat_template_id = tokenizer.encode(post_chat_template)
    return prompt_id + post_chat_template_id + response_id
