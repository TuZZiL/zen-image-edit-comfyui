"""ComfyUI nodes: Qwen-Image-2.1 driven by Qwen3.5-0.8B + the text_fusion adapter.

Drop-in for the official `TextEncodeQwenImage21`: same prompt template, same conditioning layout
(the system turn is dropped, the vision slots are removed and reported as `image_slots`), but the
17.5 GB Qwen3-VL-8B is replaced by Qwen3.5-0.8B (1.7 GB) plus the 158M adapter from
https://huggingface.co/AiArtLab/zen-image-edit.

    ZenImage21AdapterLoader  -> Qwen3.5-0.8B + processor + fusion weights (cached)
    ZenImage21TextEncode     -> prompt (+ reference images) -> CONDITIONING + latent

The DiT stays stock: its `txt_in` consumes exactly the tensor the fusion produces, so no unet
weights need patching — point `UNETLoader` at the plain Qwen-Image-2.1 model.
"""
import glob
import json
import math
import os

import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io

import fusion_lib

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

_CACHE = {}


def _device():
    try:
        return comfy.model_management.get_torch_device()
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fusion_config(meta):
    """Fusion config out of safetensors metadata.

    Understands both the adapter file (`adapter_v12.safetensors`: comma-separated `student_layers`,
    `attn_max_len`, no `mixer_ffn`/`drop_idx`) and a packed `text_fusion` file (JSON list, `max_len`).
    """
    cfg = {}
    for key, value in (meta or {}).items():
        if key == "student_layers":
            cfg[key] = [int(i) for i in str(value).strip("[]").replace(" ", "").split(",") if i]
        elif key in ("in_dim", "out_dim", "hidden", "proj_layers", "n_slices", "attention",
                     "attn_dim", "attn_heads", "max_len", "attn_max_len", "mixer", "mixer_heads",
                     "mixer_ffn", "drop_idx"):
            cfg[key] = int(value)
        elif key == "norm":
            cfg["norm"] = str(value)
    if "attn_max_len" in cfg:
        cfg.setdefault("max_len", cfg.pop("attn_max_len"))
    cfg.setdefault("max_len", 512)
    cfg.setdefault("mixer_ffn", 2)
    cfg.setdefault("drop_idx", 14)          # system-prompt prefix the DiT expects to be gone
    missing = [k for k in ("in_dim", "out_dim", "hidden", "proj_layers", "norm", "n_slices",
                           "student_layers") if k not in cfg]
    if missing:
        raise ValueError(f"adapter metadata has no {missing}; is this really an adapter file?")
    return cfg


def _load_fusion(source):
    """Fusion weights + their config, from a standalone adapter .safetensors (config in metadata) or
    from a model folder's `transformer/` (config in config.json, weights under `text_fusion.`)."""
    from safetensors import safe_open

    state = {}
    if os.path.isdir(source):
        with open(os.path.join(source, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)["text_fusion_config"]
        # a config.json holds typed values, metadata holds strings: one normalisation for both
        config = {k: (v if k == "norm" else
                      [int(i) for i in v] if k == "student_layers" else int(v))
                  for k, v in config.items()}
        prefix = "text_fusion."
        files = sorted(glob.glob(os.path.join(source, "*.safetensors")))
    else:
        with safe_open(source, framework="pt") as handle:
            config = _fusion_config(handle.metadata())
        prefix = ""
        files = [source]
    for shard in files:
        # per-tensor reads: only the adapter tensors, never the 14 GB DiT
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key[len(prefix):]] = handle.get_tensor(key)
    return config, state


class ZenImage21Adapter:
    """Qwen3.5-0.8B (frozen) plus the `text_fusion` block, loaded once per configuration."""

    def __init__(self, model_folder, text_encoder, adapter_file, dtype_name):
        from transformers import AutoProcessor, AutoTokenizer, Qwen3_5ForConditionalGeneration

        dtype = torch.float16 if dtype_name == "fp16" else torch.bfloat16
        device = _device()

        local = bool(model_folder) and os.path.isdir(os.path.join(model_folder, "text_encoder"))
        encoder = os.path.join(model_folder, "text_encoder") if local else text_encoder
        processor_path = os.path.join(model_folder, "processor") if local else text_encoder
        tokenizer_path = os.path.join(model_folder, "tokenizer") if local else text_encoder
        source = adapter_file or (os.path.join(model_folder, "transformer") if local else "")
        if not source:
            raise ValueError("set `adapter_file` (adapter_v12.safetensors) or `model_folder` "
                             "(a zen-image-edit checkout)")
        config, state = _load_fusion(source)

        fusion = fusion_lib.build_fusion(**config)
        fusion.load_state_dict(state)
        self.fusion = fusion.to(device, dtype).eval()
        self.dtype = dtype

        self.student = Qwen3_5ForConditionalGeneration.from_pretrained(encoder, dtype=dtype).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.layers = list(config["student_layers"])
        self.expect_drop = int(config["drop_idx"])
        self.im_start_id = int(self.tokenizer.convert_tokens_to_ids("<|im_start|>"))
        self.image_pad_id = int(self.tokenizer.convert_tokens_to_ids("<|image_pad|>"))
        print(f"[zen-image-edit] adapter ready: encoder={encoder}, layers={self.layers}, "
              f"dtype={dtype_name}", flush=True)

    @torch.no_grad()
    def encode(self, text, images, sizes):
        """text + reference images -> fused condition on the FULL sequence, plus the token ids."""
        if not text.strip():
            text = " "  # Qwen has no BOS, an empty string would leave the encoder with nothing
        refs = " ".join(f"<image{i + 1}>{VISION_BLOCK}" for i in range(len(images)))
        prompt = T2I_TEMPLATE.replace("{}", refs + "{}", 1).format(text) if refs else T2I_TEMPLATE.format(text)

        kwargs = {"text": [prompt], "padding": True, "padding_side": "right", "return_tensors": "pt"}
        if images:
            kwargs["images"] = images
        inputs = self.processor(**kwargs).to(self.student.device)

        forward = {"input_ids": inputs.input_ids, "attention_mask": inputs.attention_mask,
                   "output_hidden_states": True}
        if images:
            if not hasattr(inputs, "pixel_values"):
                raise ValueError("the processor returned no pixel_values for the reference images")
            grid = inputs.image_grid_thw
            got = [int(g[1] * g[2]) // 4 for g in grid]           # one vision slot per 2x2 latents
            want = [(w // 32) * (h // 32) for w, h in sizes]
            if got != want:
                raise ValueError(
                    f"vision slots {got} do not match the reference latents {want}: the encoder would "
                    f"splice the images at the wrong positions (processor resized differently)")
            forward["pixel_values"] = inputs.pixel_values.to(self.dtype)
            forward["image_grid_thw"] = grid
        if hasattr(inputs, "mm_token_type_ids"):
            forward["mm_token_type_ids"] = inputs.mm_token_type_ids

        hidden = self.student(**forward).hidden_states
        x = torch.stack([hidden[i] for i in self.layers], dim=1)          # (B, K, L, d)
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)     # (B, L, K*d)
        cond = self.fusion(x.to(self.dtype), inputs.attention_mask.bool())
        return cond, inputs.input_ids[0].tolist()


def load_adapter(model_folder, dtype_name, text_encoder="models/text_encoders/qwen3.5_0.8b",
                 adapter_file=""):
    key = (os.path.abspath(model_folder or ""), text_encoder, adapter_file, dtype_name)
    if key not in _CACHE:
        _CACHE[key] = ZenImage21Adapter(model_folder, text_encoder, adapter_file, dtype_name)
    return _CACHE[key]


def _condition(adapter, text, images, sizes, ref_latents, keep_vision):
    """Same post-processing as the stock Qwen Image 2.1 encoder: drop the system turn, then the
    vision slots, and report where each reference image goes."""
    cond, ids = adapter.encode(text, images, sizes)
    ids = torch.tensor(ids)
    starts = (ids == adapter.im_start_id).nonzero().flatten().tolist()
    drop = starts[1] if len(starts) > 1 else 0          # everything before the user turn
    if drop != adapter.expect_drop:
        raise ValueError(f"tokenizer gives {drop} prefix tokens, the checkpoint was trained with "
                         f"{adapter.expect_drop}: the condition would be shifted")
    keep = torch.ones(ids.numel(), dtype=torch.bool)
    keep[:drop] = False

    slots = []
    if not keep_vision:
        runs = []
        for i in (ids == adapter.image_pad_id).nonzero().flatten().tolist():
            if runs and i == runs[-1][0] + runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([i, 1])
        if len(runs) != len(images):
            raise ValueError(f"{len(runs)} vision slots for {len(images)} reference images")
        for start, size in runs:
            slots.append(int(keep[:start].sum()))
            keep[start:start + size] = False

    extras = {}
    if slots:
        extras["image_slots"] = slots
    if ref_latents:
        extras["reference_latents"] = ref_latents
    return [[cond[:, keep.to(cond.device)], extras]]


class ZenImage21AdapterLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZenImage21AdapterLoader",
            display_name="Zen Image Edit — Adapter Loader (Qwen3.5-0.8B)",
            category="model/conditioning/qwen image",
            inputs=[
                io.String.Input("model_folder", default="",
                                tooltip="Optional: a zen-image-edit checkout (text_encoder/, processor/, "
                                        "tokenizer/, transformer/). Leave empty to load the encoder by "
                                        "its Hugging Face id and pass `fusion_file`."),
                io.String.Input("adapter_file", default="",
                                tooltip="Path to adapter_v12.safetensors (0.6 GB, config in metadata). "
                                        "Required when `model_folder` is empty."),
                io.String.Input("text_encoder", default="models/text_encoders/qwen3.5_0.8b",
                                tooltip="Text encoder id or path, used when `model_folder` is empty."),
                io.Combo.Input("dtype", options=["bf16", "fp16"], default="bf16",
                               tooltip="Keep bf16 to match the DiT; fp16 reproduces the diffusers build."),
            ],
            outputs=[io.Custom("ZEN_ADAPTER").Output(display_name="adapter")],
        )

    @classmethod
    def execute(cls, model_folder="", adapter_file="", text_encoder="models/text_encoders/qwen3.5_0.8b",
                dtype="bf16") -> io.NodeOutput:
        return io.NodeOutput(load_adapter(model_folder, dtype, text_encoder, adapter_file))


class ZenImage21TextEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="ZenImage21TextEncode",
            display_name="Zen Image Edit — Text Encode (Qwen3.5-0.8B)",
            category="model/conditioning/qwen image",
            inputs=[
                io.Custom("ZEN_ADAPTER").Input("adapter"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", multiline=True, dynamic_prompts=True),
                io.Vae.Input("vae", optional=True),
                io.Int.Input("resolution", default=1024, min=0, max=4096, step=32,
                             tooltip="Reference images are resized to about resolution x resolution "
                                     "pixels at multiples of 32. 0 keeps their own size. The VAE is "
                                     "required for reference images: the DiT takes them as latents."),
                io.Int.Input("width", default=0, min=0, max=4096, step=32,
                             tooltip="Canvas width, 0 = take it from the first reference image. Keep it "
                                     "close to that image's resized size or the edit can shift."),
                io.Int.Input("height", default=0, min=0, max=4096, step=32,
                             tooltip="Canvas height, 0 = take it from the first reference image."),
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplateNames(
                        io.Image.Input("image"),
                        names=[f"image_{i}" for i in range(1, 17)],
                        min=0,
                    ),
                    tooltip="Reference images, seen by the text encoder and spliced in as VAE latents.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent",
                                 tooltip="Empty latent on the first reference image's size."),
            ],
        )

    @classmethod
    def execute(cls, adapter, prompt, negative_prompt, vae=None, resolution=1024, width=0, height=0,
                images: io.Autogrow.Type = None, **kwargs) -> io.NodeOutput:
        canvas_w, canvas_h = width, height      # the loop below reuses `width`/`height` for the refs
        # the loader may hand the expanded autogrow inputs over as separate `image_1`, `image_2`, ...
        # keyword arguments instead of the `images` dict, accept both
        images = dict(images or {})
        images.update({k: v for k, v in kwargs.items() if k.startswith("image_")})
        ref_latents, refs, sizes = [], [], []
        images = images or {}
        latent_w = latent_h = resolution or 1024
        for name in sorted(images, key=lambda n: int(n.rsplit("_", 1)[-1])):
            image = images[name]
            if image is None:
                continue
            samples = image[:1].movedim(-1, 1).float()
            if resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                width = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                height = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                width, height = round(samples.shape[3] / 32) * 32, round(samples.shape[2] / 32) * 32
            width, height = max(32, width), max(32, height)
            if (width, height) == (samples.shape[3], samples.shape[2]):
                scaled = image[:1]
            else:
                scaled = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled").movedim(1, -1)
            if not refs:
                latent_w, latent_h = width, height
            rgb = scaled[:, :, :, :3]
            if scaled.shape[-1] > 3:  # the vision tower sees alpha over white, the VAE keeps all four
                rgb = rgb * scaled[:, :, :, 3:] + (1.0 - scaled[:, :, :, 3:])
            if vae is None:
                raise ValueError("reference images need the VAE: the DiT splices them in as latents, "
                                 "and this adapter only predicts text positions")
            ref_latents.append(vae.encode(scaled))
            refs.append(_to_pil(rgb))
            sizes.append((width, height))

        if canvas_w >= 32 and canvas_h >= 32:
            latent_w, latent_h = canvas_w, canvas_h
        positive = _condition(adapter, prompt, refs, sizes, ref_latents, keep_vision=vae is None)
        negative = _condition(adapter, negative_prompt or "", refs, sizes, ref_latents, keep_vision=vae is None)
        latent = torch.zeros([1, 64, latent_h // 16, latent_w // 16],
                             device=comfy.model_management.intermediate_device())
        return io.NodeOutput(positive, negative, {"samples": latent})


def _to_pil(rgb):
    from PIL import Image

    array = (rgb[0].detach().cpu().float().clamp(0, 1).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(array)


class ZenImageEditExtension(ComfyExtension):
    @override
    async def get_node_list(cls) -> list[type[io.ComfyNode]]:
        return [ZenImage21AdapterLoader, ZenImage21TextEncode]


async def comfy_entrypoint() -> ZenImageEditExtension:
    return ZenImageEditExtension()
