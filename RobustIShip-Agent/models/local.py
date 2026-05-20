"""Local model runner (Gemma on CPU/GPU)."""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


class LocalModel:
    def __init__(self, model_id: str, device: str = "cpu", quant: str = "fp16",
                 max_memory_gib: float = None, max_vram_gib: float = None, num_threads: int = None):
        self.model_id = model_id
        self.device = device
        self.quant = quant
        self.max_memory_gib = max_memory_gib
        self.max_vram_gib = max_vram_gib
        self.model = None
        self.tokenizer = None
        self.last_response = None

        if num_threads is not None:
            print(f"🧵 Limiting PyTorch to {num_threads} CPU threads")
            torch.set_num_threads(num_threads)

    def load(self):
        print(f"🧠 Loading model: {self.model_id}")
        if self.device == "cuda" and torch.cuda.is_available():
            device_map = "auto"
            print("   🚀 Using GPU (CUDA)")
            self.max_memory = {0: f"{self.max_vram_gib:.0f}GiB"} if self.max_vram_gib else None
        elif self.device == "auto":
            device_map = "auto"
            self.max_memory = None
        else:
            device_map = "cpu"
            print("   🖥️ Using CPU")
            self.max_memory = None

        model_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True, "device_map": device_map}
        if self.max_memory:
            model_kwargs["max_memory"] = self.max_memory

        if self.quant == "4bit" and BITSANDBYTES_AVAILABLE:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            print("   📊 4-bit quantization (~6GB)")
        elif self.quant == "8bit" and BITSANDBYTES_AVAILABLE:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            print("   📊 8-bit quantization (~16GB)")
        elif self.quant == "fp16":
            model_kwargs["torch_dtype"] = torch.float16
            print("   📊 FP16 (~16GB)")
        else:
            model_kwargs["torch_dtype"] = torch.float32
            print("   📊 FP32 (~32GB)")

        if self.device == "cpu" and self.max_memory_gib:
            model_kwargs["max_memory"] = {"cpu": f"{self.max_memory_gib:.0f}GiB"}

        hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=hf_token, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, token=hf_token, **model_kwargs)
        self.model.eval()
        print(f"   ✅ Loaded ({self.model.num_parameters()/1e9:.1f}B params)")

    def generate(self, messages: list, max_tokens: int = 256, temperature: float = 0.0) -> str:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        self.last_response = response
        return response.strip()