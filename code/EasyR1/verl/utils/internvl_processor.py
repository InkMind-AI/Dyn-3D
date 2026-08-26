"""Minimal InternVL processor used by EasyR1's dataset and FSDP workers."""

from __future__ import annotations

from typing import Any, Optional

import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoTokenizer, BatchEncoding


IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INTERNVL_SYSTEM_MESSAGE = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、"
    "清华大学及多家合作单位联合开发的多模态大语言模型。"
)


class InternVLImageProcessor:
    """Create the pixel tensors expected by the remote-code InternVL model."""

    def __init__(self, input_size: int):
        self.input_size = int(input_size)
        self.transform = T.Compose(
            [
                T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
                T.Resize((self.input_size, self.input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, images=None, videos=None, return_tensors=None, **kwargs):
        if videos:
            raise NotImplementedError("InternVL EasyR1 adapter currently supports image-frame inputs only.")
        images = list(images or [])
        pixels = [self.transform(image if isinstance(image, Image.Image) else Image.open(image)) for image in images]
        if not pixels:
            return BatchEncoding({})
        return BatchEncoding(
            {
                "pixel_values": torch.stack(pixels),
                "image_flags": torch.ones((len(pixels), 1), dtype=torch.long),
            },
            tensor_type=return_tensors,
        )


class InternVLEasyR1Processor:
    """Expose InternVL's multi-image prompt convention through EasyR1's API."""

    model_input_names = ["input_ids", "attention_mask", "pixel_values", "image_flags"]
    image_token = IMG_CONTEXT_TOKEN

    def __init__(self, model_path: str, trust_remote_code: bool = True, use_fast: bool = True, **kwargs):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code, use_fast=use_fast
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        image_size = int(config.force_image_size or config.vision_config.image_size)
        patch_size = int(config.vision_config.patch_size)
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio**2))
        self.image_processor = InternVLImageProcessor(image_size)

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        parts = []
        for item in content or []:
            if item.get("type") == "image":
                parts.append("<image>\n")
            elif item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=False, **kwargs):
        single = not conversation or isinstance(conversation[0], dict)
        conversations = [conversation] if single else conversation
        rendered = []
        for messages in conversations:
            normalized = []
            if not messages or messages[0].get("role") != "system":
                normalized.append({"role": "system", "content": INTERNVL_SYSTEM_MESSAGE})
            for message in messages:
                normalized.append(
                    {"role": message.get("role"), "content": self._content_to_text(message.get("content"))}
                )
            rendered.append(
                self.tokenizer.apply_chat_template(
                    normalized, tokenize=False, add_generation_prompt=add_generation_prompt
                )
            )
        if not tokenize:
            return rendered[0] if single else rendered
        encoded = self(images=None, text=rendered, **kwargs)
        return encoded["input_ids"][0] if single else encoded["input_ids"]

    def __call__(
        self,
        images=None,
        text=None,
        padding=True,
        return_tensors=None,
        add_special_tokens=False,
        **kwargs,
    ):
        texts = [text] if isinstance(text, str) else list(text or [])
        if images is None:
            image_batches = [[] for _ in texts]
        elif len(texts) == 1 and (not images or not isinstance(images[0], (list, tuple))):
            image_batches = [list(images)]
        else:
            image_batches = [list(batch or []) for batch in images]
        if len(texts) != len(image_batches):
            raise ValueError(f"InternVL text/image batch mismatch: {len(texts)} != {len(image_batches)}")

        replacement = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token + IMG_END_TOKEN
        expanded = []
        for value, batch in zip(texts, image_batches):
            if value.count("<image>") != len(batch):
                raise ValueError(
                    f"InternVL prompt/image mismatch: prompt={value.count('<image>')} images={len(batch)}"
                )
            for _ in batch:
                value = value.replace("<image>", replacement, 1)
            expanded.append(value)
        return self.tokenizer(
            expanded,
            padding=padding,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        return self.tokenizer.save_pretrained(save_directory, **kwargs)
