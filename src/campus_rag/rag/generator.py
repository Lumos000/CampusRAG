from __future__ import annotations

import os
import re
from pathlib import Path


class ContextBudgetError(ValueError):
    """Refuse oversized evidence instead of silently truncating its conditions."""


class TransformersGenerator:
    def __init__(self, model_path: str | Path, max_context_tokens: int = 7168):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()
        self.max_position_tokens = self.model.config.max_position_embeddings
        self.max_context_tokens = min(max_context_tokens, self.max_position_tokens)

    def generate(self, prompt: str, thinking: bool = False, temperature: float = 0.0, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": prompt}]
        kwargs = dict(tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            inputs = self.tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
        except TypeError:
            inputs = self.tokenizer.apply_chat_template(messages, **kwargs)
        input_tokens = inputs['input_ids'].shape[-1]
        if input_tokens > self.max_context_tokens or input_tokens + max_new_tokens > self.max_position_tokens:
            raise ContextBudgetError('输入超过上下文预算，或输出预留超过模型位置上限')
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            generation["temperature"] = temperature
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        tokens = output[0][inputs["input_ids"].shape[-1] :]
        text = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return re.sub(r'^.*?</think>', '', text, flags=re.DOTALL).strip()

