"""Language engine wrapper for vLLM-based text and structured generation.

Provides a `LanguageEngine` class that initializes a vLLM model and tokenizer,
and exposes helpers for single/batch responses, including JSON-schema (Pydantic)
constrained outputs that are constructed as Pydantic model instances. The
module also includes a runnable example section demonstrating typical usage
patterns.
"""

from vllm import LLM, SamplingParams
from transformers import AutoConfig, AutoTokenizer
from vllm.sampling_params import StructuredOutputsParams
from pydantic import BaseModel

VLLM_CONFIG = {
    "model_path": "Qwen/Qwen3-32B",         # allenai/Olmo-3-7B-Instruct or Qwen/Qwen3-8B 
    "max_model_len": None,                  # Optional: specify max model length if known, else inferred
    "tensor_parallel_size": 2,              # Use >1 for multi-GPU inference
    "gpu_memory_utilization": 0.90,         # Fraction of GPU memory to use (e.g., 0.90 for 90%)
    "dtype": "bfloat16",                    # "float16" or "bfloat16"
    "quantization": None,                   # vLLM quantization mode, e.g. "awq", or None
    "kv_cache_dtype": "auto",               # KV cache dtype, e.g. "fp8" or "auto"
    "enforce_eager": True,                  # Keep existing default behavior unless overridden
    "temperature": 0.1,                     # Sampling temperature
    "top_p": 0.95,                          # Nucleus sampling top-p
    "max_tokens": 8000,                     # Maximum tokens to generate
    "yarn_rope_scaling": False,              # Enable YaRN rope scaling
    "yarn_factor": 4.0,                     # Scaling factor for YaRN rope scaling
    "native_ctx_length": None,              # Optional: specify native context length if known, just needed for YaRN and for other models outside of the Qwen3 family
    "enable_thinking": False,                # Whether to use the thinking mode in the chat template
    "tokenizer_mode": "auto",                # vLLM tokenizer mode, e.g. "auto" or "mistral"
}

class LanguageEngine:
    """Language engine constructed around vLLM for simple and structured generations.

    This class initializes a vLLM model and tokenizer, and exposes helpers for
    single/batch generation, including constrained outputs that are
    used to construct Pydantic model instances, with optional "thinking" mode.
    """

    def __init__(self, config=VLLM_CONFIG):
        """Initialize the language engine and load model/tokenizer.

        Args:
            config (dict): vLLM and sampling configuration. Expected keys
                include `model_path`, `tensor_parallel_size`,
                `gpu_memory_utilization`, `max_model_len`, `dtype`,
                `temperature`, `top_p`, `max_tokens`, and optionally `yarn_rope_scaling` and `yarn_factor`.

        Side Effects:
            Loads the model weights and tokenizer into memory and prepares
            default sampling parameters.
        """
        self.config = config

        self.model = LLM(
            model=self.config["model_path"],
            tensor_parallel_size=self.config["tensor_parallel_size"],
            gpu_memory_utilization=self.config["gpu_memory_utilization"],
            max_model_len=self.config["max_model_len"],
            dtype=self.config["dtype"],
            quantization=self.config.get("quantization"),
            kv_cache_dtype=self.config.get("kv_cache_dtype", "auto"),
            trust_remote_code=True,
            enforce_eager=self.config.get("enforce_eager", True),
            tokenizer_mode=self.config.get("tokenizer_mode", "auto"),
            hf_overrides=self.__construct_yarn_config() if self.config.get("yarn_rope_scaling", False) else None
        )

        # Some models (e.g., Ministral) may not expose a tokenizer class that
        # AutoTokenizer can import in older transformers builds.
        self.tokenizer = None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model_path"],
                trust_remote_code=True
            )
        except Exception as exc:
            print(
                "AutoTokenizer load failed; trying vLLM tokenizer fallback. "
                f"Model={self.config['model_path']} Error={exc}"
            )
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
            max_tokens=self.config["max_tokens"]
        )

    @staticmethod
    def __fallback_prompt_from_messages(messages):
        lines = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper()
            content = str(message.get("content", "")).strip()
            lines.append(f"{role}: {content}")
        lines.append("ASSISTANT:")
        return "\n\n".join(lines)

    def __apply_chat_template(self, messages, enable_thinking=False, continue_final_message=None):
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if apply_chat_template is None:
            return self.__fallback_prompt_from_messages(messages)

        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": enable_thinking,
        }
        if continue_final_message is not None:
            kwargs["continue_final_message"] = continue_final_message

        try:
            return apply_chat_template(messages, **kwargs)
        except TypeError:
            # Drop tokenizer-specific kwargs for models that do not accept them.
            kwargs.pop("enable_thinking", None)
            kwargs.pop("continue_final_message", None)
            try:
                return apply_chat_template(messages, **kwargs)
            except TypeError:
                return apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __construct_yarn_config(self):
        """Construct YaRN rope scaling configuration for the model."""
        # Load the config in a lightweight way
        config = AutoConfig.from_pretrained(self.config["model_path"], trust_remote_code=True)

        # 1. Check if the value is explicitly hidden in 'rope_scaling'
        try:
            native_ctx = getattr(config, "rope_scaling", {}).get("original_max_position_embeddings")
        except AttributeError:
            native_ctx = None

        # 2. Use the architectural formula for Qwen3: max_pos = native_ctx + thinking_buffer
        # For Qwen3, the thinking buffer is standardized at 8,192 tokens.
        if native_ctx is None:
            max_pos = config.max_position_embeddings
            thinking_buffer = 8192
            
            # Verify if the current max_pos matches the 40k buffered total
            if max_pos == 40960 and 'Qwen3' in self.config["model_path"]:
                # Safe to assume the thinking buffer is present
                native_ctx = max_pos - thinking_buffer
            else:
                if self.config.get("native_ctx_length") is not None:
                    native_ctx = self.config["native_ctx_length"]
                elif self.config["max_model_len"] is not None:
                    native_ctx = self.config["max_model_len"]
                else:
                    raise ValueError(
                        "Unable to determine native context length for YaRN rope scaling. "
                        "Please specify 'native_ctx_length' or 'max_model_len' in the config."
                    )

        # print(f"Native Context Length: {native_ctx}")
        original_max_position_embeddings = native_ctx
        # print(f"The maximum position embedding for {self.config["model_path"]} is: {original_max_position_embeddings}")

        factor = self.config.get("yarn_factor", 4.0)
        yarn_config = {
            "rope_theta": 1000000,
            "rope_parameters": {
                "rope_type": "yarn",
                "factor": factor,
                "original_max_position_embeddings": original_max_position_embeddings,
            },
            "max_model_len": int(original_max_position_embeddings * factor),
        }
        return yarn_config

    def generate(self, user_prompt, enable_thinking=False, system_prompt=None):
        """Generate a response for a single user prompt.

        Args:
            user_prompt (str): The user message to send to the model.
            enable_thinking (bool): Whether to enable the model's thinking
                mode in the chat template.
            system_prompt (str | None): Optional system message to prepend.

        Returns:
            str: The model's response text (stripped).
        """
        # PROMPT CONSTRUCTION
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        full_prompt = self.__apply_chat_template(messages, enable_thinking=enable_thinking)
        
        # INFERENCE
        outputs = self.model.generate([full_prompt], self.sampling_params, use_tqdm=False)
        
        # Clean Output
        raw_text = outputs[0].outputs[0].text.strip()

        return raw_text
    
    def generate_in_batch(self, user_prompts, enable_thinking=False, system_prompt=None):
        """Generate responses for a batch of user prompts.

        Args:
            user_prompts (list[str]): A list of user messages.
            enable_thinking (bool): Whether to enable thinking mode in the
                chat template.
            system_prompt (str | None): Optional system message to prepend.

        Returns:
            list[str]: A list of response texts aligned with inputs.
        """
        full_prompts = []
        for user_prompt in user_prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})

            full_prompt = self.__apply_chat_template(messages, enable_thinking=enable_thinking)
            full_prompts.append(full_prompt)

        outputs = self.model.generate(full_prompts, self.sampling_params, use_tqdm=False)

        results = []
        for output in outputs:
            raw_text = output.outputs[0].text.strip()
            results.append(raw_text)

        return results
    
    def generate_structured(self, user_prompt, pydantic_model, enable_thinking=False, system_prompt=None):
        """Generate a JSON response and construct a Pydantic model.

        Args:
            user_prompt (str): The user message to send to the model.
            pydantic_model (type[BaseModel]): Pydantic model class used to
                derive the JSON schema and validate the output.
            enable_thinking (bool): Whether to enable the model's thinking
                mode before structured generation.
            system_prompt (str | None): Optional system message to prepend.

        Returns:
            BaseModel: An instance of `pydantic_model` constructed from the
                generated JSON.
        """
        raw_result = self.__generate_structured(
            user_prompt,
            pydantic_model.model_json_schema(),
            enable_thinking,
            system_prompt,
        )
        try:
            return pydantic_model.model_validate_json(raw_result)
        except Exception as exc:
            repaired = self.__strip_control_chars(raw_result)
            if repaired != raw_result:
                try:
                    return pydantic_model.model_validate_json(repaired)
                except Exception:
                    pass
            print("Structured parse failed for single prompt.")
            print(f"Error: {exc}")
            print("Prompt:")
            print(user_prompt)
            print("Raw output:")
            print(raw_result)
            return None

    def generate_structured_in_batch(self, user_prompts, pydantic_model, enable_thinking=False, system_prompt=None):
        """Generate structured JSON responses and construct models in batch.

        Args:
            user_prompts (list[str]): A list of user messages.
            pydantic_model (type[BaseModel]): Pydantic model class used to
                derive the JSON schema and validate all outputs.
            enable_thinking (bool): Whether to enable the model's thinking
                mode before structured generation.
            system_prompt (str | None): Optional system message to prepend.

        Returns:
            list[BaseModel]: List of pydantic model instances constructed from the
                generated JSON outputs.
        """
        json_schema = pydantic_model.model_json_schema()
        raw_results = self.__generate_structured_in_batch(user_prompts, json_schema, enable_thinking, system_prompt)
        outputs = []
        for idx, result in enumerate(raw_results):
            try:
                outputs.append(pydantic_model.model_validate_json(result))
            except Exception as exc:
                repaired = self.__strip_control_chars(result)
                if repaired != result:
                    try:
                        outputs.append(pydantic_model.model_validate_json(repaired))
                        print(f"Structured parse recovered in batch by stripping control chars. Index: {idx}")
                        continue
                    except Exception:
                        pass
                print("Structured parse failed in batch.")
                print(f"Index: {idx}")
                print(f"Error: {exc}")
                print("Prompt:")
                print(user_prompts[idx])
                print("Raw output:")
                print(result)
                outputs.append(None)
        return outputs

    def generate_structured_in_batch_with_models(self, user_prompts, pydantic_models, enable_thinking=False, system_prompt=None):
        """Generate structured JSON responses and construct per-item models.

        Args:
            user_prompts (list[str]): A list of user messages.
            pydantic_models (list[type[BaseModel]]): A list of Pydantic model
                classes, one per prompt.
            enable_thinking (bool): Whether to enable the model's thinking
                mode before structured generation.
            system_prompt (str | None): Optional system message to prepend.

        Returns:
            list[BaseModel]: List of pydantic model instances constructed from the
                generated JSON outputs and aligned with inputs.

        Raises:
            ValueError: If `user_prompts` and `pydantic_models` differ in
                length.
        """
        if len(user_prompts) != len(pydantic_models):
            raise ValueError("user_prompts and pydantic_models must have the same length")

        raw_results = self.__generate_structured_in_batch_with_models(
            user_prompts, pydantic_models, enable_thinking, system_prompt
        )
        outputs = []
        for idx, (model, result) in enumerate(zip(pydantic_models, raw_results)):
            try:
                outputs.append(model.model_validate_json(result))
            except Exception as exc:
                repaired = self.__strip_control_chars(result)
                if repaired != result:
                    try:
                        outputs.append(model.model_validate_json(repaired))
                        print(f"Structured parse recovered in mixed batch by stripping control chars. Index: {idx}")
                        continue
                    except Exception:
                        pass
                raise ValueError(f"Structured parse failed in mixed batch at index {idx}: {exc}") from exc
        return outputs

    @staticmethod
    def __strip_control_chars(text: str) -> str:
        """
        Remove raw control chars (U+0000-U+001F) that can invalidate JSON strings.
        """
        return "".join(ch for ch in text if ord(ch) >= 32)

    def __generate_structured(self, user_prompt, json_schema, enable_thinking=False, system_prompt=None):
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        # If thinking is enabled, we will need to first generate the reasoning steps
        if enable_thinking:
            full_prompt = self.__apply_chat_template(messages, enable_thinking=True)
            thinking_outputs = self.model.generate([full_prompt], self.sampling_params, use_tqdm=False)
            reasoning_steps = thinking_outputs[0].outputs[0].text.strip()
            reasoning_steps = reasoning_steps.split("</think>")[0].strip() + "</think>"  # Extract the reasoning part and reconstruct
            full_prompt = full_prompt + reasoning_steps + "\n"
        else:
            full_prompt = self.__apply_chat_template(
                messages,
                enable_thinking=False,
                continue_final_message=False,
            )

        structured_outputs_params = StructuredOutputsParams(json=json_schema)

        structured_params = SamplingParams(
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_tokens"],
            structured_outputs=structured_outputs_params
        )

        outputs = self.model.generate([full_prompt], structured_params, use_tqdm=False)
        raw_text = outputs[0].outputs[0].text.strip()
        return raw_text

    def __generate_structured_in_batch(self, user_prompts, json_schema, enable_thinking=False, system_prompt=None):
        messages_list = []
        for user_prompt in user_prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages_list.append(messages)

        if enable_thinking:
            thinking_prompts = [
                self.__apply_chat_template(messages, enable_thinking=True)
                for messages in messages_list
            ]
            thinking_outputs = self.model.generate(thinking_prompts, self.sampling_params, use_tqdm=False)
            reasoning_steps = []
            for output in thinking_outputs:
                step = output.outputs[0].text.strip()
                step = step.split("</think>")[0].strip() + "</think>"
                reasoning_steps.append(step)

            full_prompts = [
                prompt + reasoning + "\n"
                for prompt, reasoning in zip(thinking_prompts, reasoning_steps)
            ]
        else:
            full_prompts = [
                self.__apply_chat_template(
                    messages,
                    enable_thinking=False,
                    continue_final_message=False,
                )
                for messages in messages_list
            ]

        structured_outputs_params = StructuredOutputsParams(json=json_schema)
        structured_params = SamplingParams(
            temperature=self.config["temperature"],
            top_p=self.config["top_p"],
            max_tokens=self.config["max_tokens"],
            structured_outputs=structured_outputs_params
        )

        outputs = self.model.generate(full_prompts, structured_params, use_tqdm=False)
        results = []
        for output in outputs:
            raw_text = output.outputs[0].text.strip()
            results.append(raw_text)
        return results

    def __generate_structured_in_batch_with_models(self, user_prompts, pydantic_models, enable_thinking=False, system_prompt=None):
        messages_list = []
        for user_prompt in user_prompts:
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": user_prompt})
            messages_list.append(messages)

        if enable_thinking:
            thinking_prompts = [
                self.__apply_chat_template(messages, enable_thinking=True)
                for messages in messages_list
            ]
            thinking_outputs = self.model.generate(thinking_prompts, self.sampling_params, use_tqdm=False)
            reasoning_steps = []
            for output in thinking_outputs:
                step = output.outputs[0].text.strip()
                step = step.split("</think>")[0].strip() + "</think>"
                reasoning_steps.append(step)

            full_prompts = [
                prompt + reasoning + "\n"
                for prompt, reasoning in zip(thinking_prompts, reasoning_steps)
            ]
        else:
            full_prompts = [
                self.__apply_chat_template(
                    messages,
                    enable_thinking=False,
                    continue_final_message=False,
                )
                for messages in messages_list
            ]

        structured_params_list = []
        for model in pydantic_models:
            json_schema = model.model_json_schema()
            structured_outputs_params = StructuredOutputsParams(json=json_schema)
            structured_params = SamplingParams(
                temperature=self.config["temperature"],
                top_p=self.config["top_p"],
                max_tokens=self.config["max_tokens"],
                structured_outputs=structured_outputs_params
            )
            structured_params_list.append(structured_params)

        outputs = self.model.generate(full_prompts, structured_params_list, use_tqdm=False)
        results = []
        for output in outputs:
            raw_text = output.outputs[0].text.strip()
            results.append(raw_text)
        return results

# Example usage (run this file directly to try the API):
if __name__ == "__main__":
    engine = LanguageEngine()

    # # 1) Single-prompt text generation (basic chat-style usage).
    # #    Shows the minimal call pattern; use this when you only need one reply.
    # prompt = "Explain the theory of relativity in simple terms."
    # response = engine.generate(prompt)
    # print("Response:", response)

    # # 2) Batch text generation to amortize overhead across multiple prompts.
    # #    Shows how to process many prompts at once; use for throughput-sensitive workloads.
    # prompts = ["What is quantum computing?", "Describe the process of photosynthesis."]
    # responses = engine.generate_in_batch(prompts)
    # for i, resp in enumerate(responses):
    #     print(f"Response {i+1}:", resp)

    # # 3) Structured generation: define a schema with Pydantic and get a model instance back.
    # #    Shows schema-constrained outputs; use when downstream code needs typed fields.
    # class CarDescription(BaseModel):
    #     brand: str
    #     model: str
    #     car_type: str

    # structured_prompt = (
    #     "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's. Think step by step which was the most iconic car and then provide the answer in JSON format."
    # )
    # structured_response = engine.generate_structured(
    #     structured_prompt, pydantic_model=CarDescription, enable_thinking=False
    # )
    # print("Structured Response:", structured_response)
    # print(f"Structured Response Type: {type(structured_response)}")

    # # 4) Structured generation with `enable_thinking=True` to allow a reasoning prefix.
    # #    Shows how to enable thinking tokens; use if the model benefits from chain-of-thought.
    # structured_response = engine.generate_structured(
    #     structured_prompt, pydantic_model=CarDescription, enable_thinking=True
    # )
    # print("Structured Response:", structured_response)
    # print(f"Structured Response Type: {type(structured_response)}")

    # # 5) Batch structured generation with a single shared schema.
    # #    Shows batch + schema together; use when many prompts share one output shape.
    # batch_structured_prompts = [
    #     "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's.",
    #     "Generate a JSON with the brand, model and car_type of a famous Japanese sports car.",
    # ]
    # batch_structured_responses = engine.generate_structured_in_batch(
    #     batch_structured_prompts, pydantic_model=CarDescription, enable_thinking=False
    # )
    # for i, resp in enumerate(batch_structured_responses):
    #     print(f"Batch Structured Response {i+1}:", resp)
    #     print(f"Batch Structured Response Type {i+1}: {type(resp)}")
    # # 6) Batch structured generation with `enable_thinking=True`.
    # #    Shows batch structured outputs with thinking; use for complex, multi-step tasks.
    # batch_structured_responses = engine.generate_structured_in_batch(
    #     batch_structured_prompts, pydantic_model=CarDescription, enable_thinking=True
    # )
    # for i, resp in enumerate(batch_structured_responses):
    #     print(f"Batch Structured Response {i+1}:", resp)
    #     print(f"Batch Structured Response Type {i+1}: {type(resp)}")

    # # 7) Batch structured generation with different schemas per prompt.
    # #    Shows per-item schemas; use when each prompt needs a different output shape.
    # class MovieDescription(BaseModel):
    #     title: str
    #     year: int
    #     genre: str

    # mixed_prompts = [
    #     "Generate a JSON with the brand, model and car_type of the most iconic car from the 90's.",
    #     "Generate a JSON with the title, year, and genre of a classic 90's sci-fi movie.",
    # ]
    # mixed_models = [CarDescription, MovieDescription]

    # mixed_responses = engine.generate_structured_in_batch_with_models(
    #     mixed_prompts, mixed_models, enable_thinking=False
    # )
    # for i, resp in enumerate(mixed_responses):
    #     print(f"Mixed Structured Response {i+1}:", resp)
    #     print(f"Mixed Structured Response Type {i+1}: {type(resp)}")
    # # 8) Mixed-schema batch generation with `enable_thinking=True`.
    # #    Shows mixed schemas plus thinking; use when prompts vary and need reasoning.
    # mixed_responses = engine.generate_structured_in_batch_with_models(
    #     mixed_prompts, mixed_models, enable_thinking=True
    # )
    # for i, resp in enumerate(mixed_responses):
    #     print(f"Mixed Structured Response (Thinking) {i+1}:", resp)
    #     print(f"Mixed Structured Response Type (Thinking) {i+1}: {type(resp)}")
    
    # # 9) Structured generation with enum-constrained fields via Pydantic.
    # #    Shows constrained values; use to restrict outputs to a safe predefined set.
    # from pydantic import Field
    # from enum import Enum
    # class CarType(str, Enum):
    #     sedan = "sedan"
    #     suv = "suv"
    #     coupe = "coupe"
    #     convertible = "convertible"
    #     hatchback = "hatchback"
    #     wagon = "wagon"
    #     van = "van"
    #     truck = "truck"

    # class CarDescriptionWithEnum(BaseModel):
    #     brand: str
    #     model: str
    #     car_type: CarType = Field(..., description="Type of the car from the predefined enum options.")

    # structured_response_enum = engine.generate_structured(
    #     structured_prompt, pydantic_model=CarDescriptionWithEnum, enable_thinking=False
    # )
    # print("Structured Response with Enum:", structured_response_enum)
    # print(f"Structured Response with Enum Type: {type(structured_response_enum)}")

    # # 10) Structured generation with enum-constrained fields and `enable_thinking=True`.
    # #     Shows enum constraints plus reasoning; use for complex tasks needing safe outputs.
    # structured_response_enum = engine.generate_structured(
    #     structured_prompt, pydantic_model=CarDescriptionWithEnum, enable_thinking=True
    # )
    # print("Structured Response with Enum (Thinking):", structured_response_enum)
    # print(f"Structured Response with Enum Type (Thinking): {type(structured_response_enum)}")

    # 11) Testing YaRN rope scaling with extended context length.
    #     Shows how to initialize the engine with YaRN scaling enabled.
    with open("tests/text_long.txt", "r") as f:
        long_text = f.read()
        # Take just the first 300,000 characters to ensure we are within the 131k token limit after tokenization, given that tokens can be shorter than characters.
        long_text = long_text[:500000] + "... (truncated for testing)"
    # long_prompt = "Summarize the following text: " + "Lorem ipsum " * 30000  # Long text to test extended context
    long_prompt = "Read the following text: " + long_text + "Forget about the text and Tell me a joke about Donald Trump"  # Long text to test extended context
    # Count tokens with the tokenizer to verify extended context handling
    tokenized = engine.tokenizer(long_prompt, return_tensors="np")
    print(f"Tokenized input length: {len(tokenized['input_ids'][0])} tokens")
    long_response = engine.generate(long_prompt, enable_thinking=True)
    print("Long Response with YaRN Scaling:", long_response)

    class BookSummary(BaseModel):
        title: str
        main_characters: list[str]
        summary: str
        major_themes: list[str]
    structured_long_response = engine.generate_structured(
        long_prompt, pydantic_model=BookSummary, enable_thinking=True
    )
    print("Structured Long Response with YaRN Scaling:", structured_long_response)
