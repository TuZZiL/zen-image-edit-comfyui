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
import gc
import glob
import json
import math
import os
from collections import OrderedDict

import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io

import fusion_lib

# This fork ships suffixed node ids so it can be installed next to the upstream
# `zen-image-edit-comfyui` in the same ComfyUI: node ids are the dictionary keys
# ComfyUI registers on, and two customs nodes claiming the same key collide —
# the second one silently replaces the first in the node menu.
# Set ZEN_NODE_SUFFIX="" to get the upstream ids back (required if you only run
# this copy, or when preparing an upstream pull request).
NODE_SUFFIX = os.environ.get("ZEN_NODE_SUFFIX", "Plus")

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

# Each cache entry pins the encoder (~1.7 GB) plus the fusion adapter (~0.6 GB) in VRAM.
# Two is enough to compare configurations; more will not fit on a 12 GB card.
_CACHE_MAX = 2
_CACHE = OrderedDict()   # LRU: load_adapter moves hits to the end and evicts the front

ENCODER_DOWNLOAD_HINT = (
    "hf download Qwen/Qwen3.5-0.8B --local-dir "
    "<ComfyUI>/models/text_encoders/qwen3.5_0.8b"
)


def resolve_adapter_path(value):
    """Resolve `adapter_file` to a real path, or None.

    `adapter_file` is handed straight to `safetensors.safe_open`, which resolves a
    relative value against the *process* working directory — not against
    `ComfyUI/models/`. Users reasonably expect the latter, so we try ComfyUI's
    registered model folders before falling back to the working directory.
    """
    if not value:
        return None
    candidate = os.path.abspath(os.path.expanduser(value))
    if os.path.isfile(candidate):
        return candidate

    if not os.path.isabs(value) and not value.startswith("."):
        roots = []
        try:  # ComfyUI only; absent in a bare test environment
            import folder_paths
            for key in ("safetensors", "text_encoders", "clip", "unet", "checkpoints", "loras"):
                try:
                    roots.extend(folder_paths.get_folder_paths(key) or [])
                except Exception:
                    continue
            roots.append(folder_paths.models_dir)
        except Exception:
            pass

        for root in roots:
            for cand in (os.path.join(root, value), os.path.join(root, "..", value)):
                cand = os.path.abspath(cand)
                if os.path.isfile(cand):
                    return cand

    return None


def _encoder_weights(directory):
    """Classify the weights sitting in an encoder directory."""
    entries = os.listdir(directory)
    has_canonical = os.path.isfile(os.path.join(directory, "model.safetensors"))
    has_bin = os.path.isfile(os.path.join(directory, "pytorch_model.bin"))
    has_index = os.path.isfile(os.path.join(directory, "model.safetensors.index.json"))
    shards = [e for e in entries if e.endswith(".safetensors") and e != "model.safetensors"]
    return has_canonical, has_bin, has_index, shards


def validate_text_encoder(value):
    """Return (encoder, processor, tokenizer) paths, or explain what is wrong.

    `from_pretrained` accepts either a directory or a Hugging Face repo id, and
    fails at three different places depending on which mistake was made. All
    three are turned into directed messages here, because the raw exceptions
    (`HFValidationError`, `OSError: Error no file named model.safetensors...`)
    name neither the cause nor the fix.
    """
    if not value or not str(value).strip():
        raise ValueError(
            "text_encoder is empty: set it to the encoder directory "
            "(e.g. <ComfyUI>/models/text_encoders/qwen3.5_0.8b) or to the "
            "Hugging Face id Qwen/Qwen3.5-0.8B"
        )

    raw = str(value).strip()
    path = os.path.expanduser(raw)

    if os.path.isdir(path):
        missing = []
        config = os.path.join(path, "config.json")
        if not os.path.isfile(config):
            missing.append("config.json")

        has_canonical, has_bin, has_index, shards = _encoder_weights(path)

        if not (has_canonical or has_bin or has_index):
            if shards:
                raise ValueError(
                    f"text_encoder directory {path} has weight shards "
                    f"({', '.join(sorted(shards)[:3])}) but no "
                    "model.safetensors.index.json. Without the index, "
                    "from_pretrained cannot see them and reports "
                    "\"no file named model.safetensors, or pytorch_model.bin\". "
                    "The download is incomplete — finish it with:\n"
                    f"  {ENCODER_DOWNLOAD_HINT}"
                )
            raise ValueError(
                f"text_encoder directory {path} contains no weights: expected "
                "model.safetensors, pytorch_model.bin or a sharded "
                "model.safetensors.index.json plus shards.\n"
                f"Download them with:\n  {ENCODER_DOWNLOAD_HINT}"
            )

        if missing:
            raise ValueError(
                f"text_encoder directory {path} is missing {missing}. "
                f"Download the full repo with:\n  {ENCODER_DOWNLOAD_HINT}"
            )
        return path, path, path

    if os.path.isfile(path) or path.endswith((".safetensors", ".bin", ".pt", ".gguf")):
        raise ValueError(
            f"text_encoder must be a DIRECTORY, but got a file: {path}\n"
            "from_pretrained() loads a folder (config.json + weights + tokenizer). "
            "A file path makes it treat the string as a Hugging Face repo id and "
            "raise HFValidationError, or fail with \"no file named model.safetensors\".\n"
            "Pass the folder that contains config.json, e.g. "
            "<ComfyUI>/models/text_encoders/qwen3.5_0.8b — or the repo id "
            "Qwen/Qwen3.5-0.8B."
        )

    # A Hugging Face repo id: exactly "owner/name", no backslashes, and whose first
    # segment is not a local directory. Do NOT test for a file extension here —
    # real ids carry dots ("Qwen/Qwen3.5-0.8B"), and splitext mistakes ".5-0.8B"
    # for one. The library downloads the id on first use.
    looks_like_id = (
        raw.count("/") == 1
        and "\\" not in raw
        and not raw.startswith(("/", ".", "~"))
        and not os.path.isdir(raw.split("/")[0])
    )
    if looks_like_id:
        return raw, raw, raw

    raise ValueError(
        f"text_encoder directory not found: {path}\n"
        "Give either a directory or a Hugging Face id (owner/name). To fetch the "
        f"encoder:\n  {ENCODER_DOWNLOAD_HINT}"
    )


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
        resolved = resolve_adapter_path(source)
        if resolved is None:
            tried = os.path.abspath(os.path.expanduser(source))
            hint = ""
            if not os.path.isabs(source):
                hint = ("\nThis is a relative path: it resolves against the process "
                        "working directory (" + os.getcwd() + "), NOT against "
                        "ComfyUI/models/. Pass an absolute path.")
            raise FileNotFoundError(
                f"adapter file not found: '{source}'\n"
                f"resolved to: {tried}\n"
                "`adapter_file` must point at the adapter_v12.safetensors file "
                "(0.64 GB, from the zen-image-edit release assets). Put it in "
                "<ComfyUI>/models/ and pass the ABSOLUTE path to it." + hint
            )
        try:
            with safe_open(resolved, framework="pt") as handle:
                config = _fusion_config(handle.metadata())
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"adapter file '{resolved}' could not be read as a safetensors "
                f"adapter: {type(exc).__name__}: {exc}\n"
                "Expected the adapter_v12.safetensors release asset (its config "
                "lives in the safetensors metadata)."
            ) from exc
        prefix = ""
        files = [resolved]
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
        if local:
            encoder = os.path.join(model_folder, "text_encoder")
            processor_path = os.path.join(model_folder, "processor")
            tokenizer_path = os.path.join(model_folder, "tokenizer")
        else:
            # directed errors instead of HFValidationError / "no file named model.safetensors"
            encoder, processor_path, tokenizer_path = validate_text_encoder(text_encoder)
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


def _release(adapter):
    """Drop one adapter out of VRAM before evicting it from the cache.

    Every entry holds ~2.3 GB. Without this, changing any of the four cache-key
    fields on a 12 GB card walks straight into OOM.
    """
    for attr in ("student", "fusion", "processor", "tokenizer"):
        held = getattr(adapter, attr, None)
        if held is None:
            continue
        try:
            held.to("cpu")
        except Exception:
            pass
    del adapter
    gc.collect()
    try:
        comfy.model_management.soft_empty_cache()
    except Exception:
        pass


def load_adapter(model_folder, dtype_name, text_encoder="models/text_encoders/qwen3.5_0.8b",
                 adapter_file=""):
    key = (os.path.abspath(model_folder or ""), text_encoder, adapter_file, dtype_name)
    cached = _CACHE.get(key)
    if cached is not None:
        _CACHE.move_to_end(key)
        return cached

    while len(_CACHE) >= _CACHE_MAX:
        old_key = next(iter(_CACHE))
        print(f"[zen-image-edit] evicting cached adapter for {old_key[-2]} "
              f"(cache limit {_CACHE_MAX})", flush=True)
        _release(_CACHE.pop(old_key))

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


# ---------------------------------------------------------------------------
# Input pickers
#
# Typing an absolute path into a widget is a relic; ComfyUI's own loaders fill a
# dropdown from `folder_paths` and offer an upload button. We do the same: scan
# the registered model folders for the adapter, and the `text_encoders/` folders
# for the encoder. Both helpers degrade to a single-entry list when ComfyUI is
# absent (tests, plain Python), because a Combo must never be empty.
# ---------------------------------------------------------------------------

_ENCODER_DEFAULT_ID = "Qwen/Qwen3.5-0.8B"


def _model_dirs() -> list[str]:
    """Registered model directories, ComfyUI's own `models/` last-resort included."""
    dirs: list[str] = []
    try:
        import folder_paths  # type: ignore
    except Exception:
        return dirs
    for key in ("diffusion_models", "checkpoints"):
        try:
            dirs.extend(folder_paths.get_folder_paths(key) or [])
        except Exception:
            pass
    try:
        base = getattr(folder_paths, "models_dir", "")
        if base:
            dirs.append(base)
    except Exception:
        pass
    seen: set[str] = set()
    out: list[str] = []
    for d in dirs:
        if d and d not in seen:
            seen.add(d)
            out.append(d)
    return out


def adapter_options() -> list[str]:
    """`.safetensors` names found in the model folders — the adapter dropdown."""
    names: set[str] = set()
    for d in _model_dirs():
        try:
            for f in os.listdir(d):
                if f.endswith((".safetensors", ".sft")):
                    names.add(f)
        except OSError:
            continue
    return sorted(names) or [""]


def encoder_options() -> list[str]:
    """Folders under `models/text_encoders/`, plus the remote id we ship by default."""
    names: set[str] = {_ENCODER_DEFAULT_ID}
    for d in _model_dirs():
        sub = os.path.join(d, "text_encoders")
        try:
            for f in os.listdir(sub):
                if os.path.isdir(os.path.join(sub, f)):
                    names.add(f)
        except OSError:
            continue
    return sorted(names)


def _upload_model():
    """`io.UploadType.model` when this ComfyUI exposes it, else None (older builds)."""
    return getattr(getattr(io, "UploadType", None), "model", None)


class ZenImage21AdapterLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=f"ZenImage21AdapterLoader{NODE_SUFFIX}",
            display_name=f"Zen Image Edit{NODE_SUFFIX and ' ' + NODE_SUFFIX} — "
                         "Adapter Loader (Qwen3.5-0.8B)",
            category="model/conditioning/qwen image",
            inputs=[
                io.String.Input("model_folder", default="", optional=True,
                                tooltip="Optional: a zen-image-edit checkout (text_encoder/, processor/, "
                                        "tokenizer/, transformer/). Leave empty to load the encoder by "
                                        "its Hugging Face id and pass `fusion_file`."),
                io.Combo.Input("adapter_file", options=adapter_options(), default="",
                               upload=_upload_model(),
                               tooltip="The adapter file (0.6 GB, config in metadata). Pick it from "
                                       "the list, or use the upload button to add it — ComfyUI puts "
                                       "uploads in models/. Required when `model_folder` is empty."),
                io.Combo.Input("text_encoder", options=encoder_options(),
                               default=_ENCODER_DEFAULT_ID,
                               tooltip="Text encoder: a folder under models/text_encoders/ "
                                       "(pick it from the list) or a Hugging Face repo id."),
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
            node_id=f"ZenImage21TextEncode{NODE_SUFFIX}",
            display_name=f"Zen Image Edit{NODE_SUFFIX and ' ' + NODE_SUFFIX} — "
                         "Text Encode (Qwen3.5-0.8B)",
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
