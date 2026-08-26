from pathlib import Path


path = Path("/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/transformers.py")
text = path.read_text()

old_init = """        model_mm_processor_kwargs = (self.info.ctx.model_config.mm_processor_kwargs
                                     or {})
        if model_mm_processor_kwargs:
            hf_processor_mm_kwargs = {
                **model_mm_processor_kwargs,
                **dict(hf_processor_mm_kwargs),
            }

        mm_items = self._to_mm_items(mm_data)
        hf_processor = self.info.get_hf_processor(**hf_processor_mm_kwargs)
"""
new_init = """        model_mm_processor_kwargs = (self.info.ctx.model_config.mm_processor_kwargs
                                     or {})
        processor_init_kwargs = dict(hf_processor_mm_kwargs)
        merged_mm_processor_kwargs = {
            **model_mm_processor_kwargs,
            **processor_init_kwargs,
        }

        mm_items = self._to_mm_items(mm_data)
        hf_processor = self.info.get_hf_processor(**processor_init_kwargs)
"""

old_apply = """             hf_processor_mm_kwargs=hf_processor_mm_kwargs,
"""
new_apply = """             hf_processor_mm_kwargs=merged_mm_processor_kwargs,
"""

old_count = """        mm_processor_kwargs = dict(hf_processor_mm_kwargs)
"""
new_count = """        mm_processor_kwargs = dict(merged_mm_processor_kwargs)
"""

if old_init not in text:
    raise SystemExit("initial processor kwargs block not found")
text = text.replace(old_init, new_init, 1)

if old_apply not in text:
    raise SystemExit("processor apply kwargs block not found")
text = text.replace(old_apply, new_apply, 1)

if old_count not in text:
    raise SystemExit("token count kwargs block not found")
text = text.replace(old_count, new_count, 1)

backup = path.with_suffix(".py.bak_easyr1_qwen3vl_processor_kwargs_fix")
if not backup.exists():
    backup.write_text(path.read_text())
path.write_text(text)
print(f"patched {path}; backup={backup}")
