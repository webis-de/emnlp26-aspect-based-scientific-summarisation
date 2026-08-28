"""Minimal vLLM language engine for the COD experiment."""

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

VLLM_CONFIG = {
    "model_path": "Qwen/Qwen3.5-9B",
    "max_model_len": None,
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.90,
    "dtype": "bfloat16",
    "quantization": None,
    "kv_cache_dtype": "auto",
    "enforce_eager": True,
    "temperature": 0.1,
    "top_p": 0.95,
    "max_tokens": 8000,
    "enable_thinking": False,
    "tokenizer_mode": "auto",
}


class LanguageEngine:
    def __init__(self, config=VLLM_CONFIG):
        self.config = config
        self.model = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=self.config["tensor_parallel_size"],
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            max_model_len=self.config.get("max_model_len"),
            dtype=self.config["dtype"],
            quantization=self.config.get("quantization"),
            kv_cache_dtype=self.config.get("kv_cache_dtype", "auto"),
            trust_remote_code=True,
            enforce_eager=self.config.get("enforce_eager", True),
            tokenizer_mode=self.config.get("tokenizer_mode", "auto"),
        )

        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_path"],
                trust_remote_code=True,
            )
        except Exception:
            if hasattr(self.model, "get_tokenizer"):
                self.tokenizer = self.model.get_tokenizer()

        if self.tokenizer is None:
            raise RuntimeError(
                "Failed to initialize tokenizer for LanguageEngine. "
                f"Model={self.config['model_path']}"
            )

        self.sampling_params = SamplingParams(
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_tokens"],
        )

    @staticmethod
    def _fallback_prompt_from_messages(messages):
        lines = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper()
            content = str(message.get("content", "")).strip()
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def _apply_chat_template(self, messages, enable_thinking=False):
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return self._fallback_prompt_from_messages(messages)

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            try:
                return apply_chat_template(messages, **kwargs)
            except TypeError:
                return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, user_prompt, enable_thinking=False, system_prompt=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        full_prompt = self._apply_chat_template(messages, enable_thinking=enable_thinking)
        outputs = self.model.generate([full_prompt], self.sampling_params, use_tqdm=False)
        return outputs[0].outputs[0].text.strip()

    def generate_in_batch(self, user_prompts, enable_thinking=False, system_prompt=None):
        full_prompts = []
        for user_prompt in user_prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            full_prompts.append(self._apply_chat_template(messages, enable_thinking=enable_thinking))

        outputs = self.model.generate(full_prompts, self.sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]