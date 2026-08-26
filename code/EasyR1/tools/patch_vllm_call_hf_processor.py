from pathlib import Path


path = Path("/usr/local/lib/python3.12/site-packages/vllm/multimodal/processing.py")
text = path.read_text()

old = """        return self.info.ctx.call_hf_processor(
            self.info.get_hf_processor(**mm_kwargs),
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )
"""
new = """        try:
            hf_processor = self.info.get_hf_processor(**mm_kwargs)
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            hf_processor = self.info.get_hf_processor()
        return self.info.ctx.call_hf_processor(
            hf_processor,
            dict(text=prompt, **mm_data),
            dict(**mm_kwargs, **tok_kwargs),
        )
"""

if old not in text:
    raise SystemExit("call_hf_processor block not found")

backup = path.with_suffix(".py.bak_easyr1_qwen3vl_call_hf_processor")
if not backup.exists():
    backup.write_text(text)
path.write_text(text.replace(old, new, 1))
print(f"patched {path}; backup={backup}")
