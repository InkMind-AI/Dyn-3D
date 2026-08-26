from pathlib import Path


path = Path("/usr/local/lib/python3.12/site-packages/vllm/model_executor/models/transformers.py")
text = path.read_text()

old = """            vision_embeddings = self.model.get_image_features(
                pixel_values,
                **{
                    k: v.flatten(0, 1)
                    for k, v in kwargs.items()
                },
            )

            if isinstance(vision_embeddings, torch.Tensor):
"""
new = """            vision_embeddings = self.model.get_image_features(
                pixel_values,
                **{
                    k: v.flatten(0, 1)
                    for k, v in kwargs.items()
                },
            )
            if isinstance(vision_embeddings, tuple):
                vision_embeddings = vision_embeddings[0]

            if isinstance(vision_embeddings, torch.Tensor):
"""

if old not in text:
    raise SystemExit("vision_embeddings block not found")

backup = path.with_suffix(".py.bak_easyr1_qwen3vl_tuple_embeddings")
if not backup.exists():
    backup.write_text(text)
path.write_text(text.replace(old, new, 1))
print(f"patched {path}; backup={backup}")
