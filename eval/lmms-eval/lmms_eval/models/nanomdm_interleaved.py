import hashlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


ROOT = Path(__file__).resolve().parents[6]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.interleaved_eval.common import generate_completion, load_hf_model_bundle
from scripts.prompt_contract import (
    build_llada_header_mapped_prompt_contract,
    contract_prompt_style,
    get_prompt_contract_stop_token_ids,
    load_prompt_contract,
    render_llada_header_mapped_generation_prompt,
    render_llada_header_mapped_message,
)


@register_model("nanomdm_interleaved")
class NanoMDMInterleaved(lmms):
    def __init__(
        self,
        pretrained: str,
        training_ckpt_dir: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        attn_implementation: str = "flash_attention_2",
        batch_size: Union[int, str] = 1,
        sample_steps: int = 256,
        inference_block_size: int = 0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        media_cache_dir: str = "",
        prompt_style: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {sorted(kwargs)}")

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size_per_gpu = int(batch_size)
        self.sample_steps = int(sample_steps)
        self.inference_block_size = int(inference_block_size)
        self.cfg_scale = float(cfg_scale)
        self.remasking = str(remasking)
        self._logged_generation_kwargs = False
        requested_prompt_style = str(prompt_style).strip().lower()
        if requested_prompt_style not in {"auto", "plain", "user_assistant", "llada_header_mapped"}:
            raise ValueError(f"Unsupported prompt_style: {requested_prompt_style}")
        self._device = torch.device(device)

        cache_root = Path(media_cache_dir).expanduser().resolve() if media_cache_dir else (ROOT / "outputs" / "benchmark" / "lmms_media_cache" / "nanomdm_interleaved")
        cache_root.mkdir(parents=True, exist_ok=True)
        self.media_cache_dir = cache_root

        resolved_model_dir = Path(pretrained).expanduser().resolve()
        self.prompt_contract: Optional[Dict[str, Any]] = load_prompt_contract(resolved_model_dir)
        self.bundle = load_hf_model_bundle(
            model_dir=resolved_model_dir,
            training_ckpt_dir=Path(training_ckpt_dir).expanduser().resolve(),
            device=device,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )
        self._model = self.bundle.model
        self._tokenizer = self.bundle.tokenizer
        if requested_prompt_style == "auto":
            self.prompt_style = contract_prompt_style(self.prompt_contract) or "user_assistant"
        else:
            self.prompt_style = requested_prompt_style
        if self.prompt_style == "llada_header_mapped":
            if contract_prompt_style(self.prompt_contract) != "llada_header_mapped":
                self.prompt_contract = build_llada_header_mapped_prompt_contract(self._tokenizer)
        elif self.prompt_style not in {"plain", "user_assistant"}:
            raise ValueError(f"Unsupported resolved prompt_style: {self.prompt_style}")

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self._model

    @property
    def device(self):
        return self._device

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("NanoMDM interleaved benchmark wrapper only supports generate_until tasks.")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[str]:
        return self.generate_until(requests)

    def _flatten_visuals(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            out: List[Any] = []
            for item in value:
                out.extend(self._flatten_visuals(item))
            return out
        return [value]

    def _persist_pil_image(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
        raw = buffer.getvalue()
        image_hash = hashlib.sha256(raw).hexdigest()
        out_path = self.media_cache_dir / f"{image_hash}.png"
        if not out_path.exists():
            out_path.write_bytes(raw)
        return str(out_path)

    def _coerce_visual_to_path(self, visual: Any) -> str:
        if isinstance(visual, Image.Image):
            return self._persist_pil_image(visual)
        if isinstance(visual, np.ndarray):
            return self._persist_pil_image(Image.fromarray(visual))
        if isinstance(visual, str):
            candidate = Path(visual).expanduser()
            if candidate.exists():
                return str(candidate.resolve())
        raise TypeError(f"Unsupported visual payload for NanoMDM interleaved wrapper: {type(visual)}")

    def _build_segments(self, context: str, visuals: Iterable[Any]) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for visual in visuals:
            visual_path = self._coerce_visual_to_path(visual)
            segments.append(
                {
                    "kind": "image",
                    "image_ref": {
                        "source": "local_path",
                        "path": visual_path,
                    },
                }
            )
        cleaned_context = str(context).replace("<image>", "").strip()
        if self.prompt_style == "llada_header_mapped":
            if not self.prompt_contract:
                raise ValueError("llada_header_mapped prompt_style requires a prompt contract.")
            system_message = str(self.prompt_contract.get("system_message", "")).strip()
            if system_message:
                segments.append(
                    {
                        "kind": "text",
                        "text": render_llada_header_mapped_message(
                            self.prompt_contract,
                            role="system",
                            content=system_message,
                            include_bos=True,
                        ),
                        "score": False,
                    }
                )
            segments.append(
                {
                    "kind": "text",
                    "text": render_llada_header_mapped_message(
                        self.prompt_contract,
                        role="user",
                        content=cleaned_context,
                        include_bos=False,
                    ),
                    "score": False,
                }
            )
            segments.append(
                {
                    "kind": "text",
                    "text": render_llada_header_mapped_generation_prompt(self.prompt_contract),
                    "score": False,
                }
            )
            return segments
        if self.prompt_style == "user_assistant":
            cleaned_context = f"User: {cleaned_context}\nAssistant: "
        segments.append(
            {
                "kind": "text",
                "text": cleaned_context,
                "score": False,
            }
        )
        return segments

    def _normalize_until(self, gen_kwargs: dict[str, Any]) -> list[str]:
        until = gen_kwargs.get("until", [])
        if isinstance(until, str):
            stops = [until]
        elif isinstance(until, list):
            stops = [str(item) for item in until if str(item)]
        else:
            stops = []
        # For llada_header_mapped checkpoints, the real chat boundary is the
        # prompt-contract eot token id, applied before text decoding in
        # generate_completion(). Task-level string stops such as "ASSISTANT:"
        # are legacy LLaVA/Vicuna stops and are left here only for compatibility.
        if self.prompt_style == "user_assistant":
            # Match the turn boundary implied by the prompt template, similar in
            # spirit to how LLaDA-V derives stop strings from its conv template.
            stops.extend(["\nUser:", "\nAssistant:", "User:", "Assistant:"])
        deduped: list[str] = []
        seen: set[str] = set()
        for stop in stops:
            if stop and stop not in seen:
                deduped.append(stop)
                seen.add(stop)
        return deduped

    def _apply_until(self, text: str, until: list[str]) -> str:
        if not until:
            return text.strip()
        cut_positions = [text.find(stop) for stop in until]
        cut_positions = [idx for idx in cut_positions if idx >= 0]
        if not cut_positions:
            return text.strip()
        return text[: min(cut_positions)].strip()

    def _resolve_generation_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_new_tokens": int(gen_kwargs.get("gen_length", gen_kwargs.get("max_new_tokens", 64))),
            "temperature": float(gen_kwargs.get("temperature", 0.0)),
            "top_p": float(gen_kwargs.get("top_p", 1.0)),
            "sample_steps": int(gen_kwargs.get("gen_steps", gen_kwargs.get("sample_steps", self.sample_steps))),
            "inference_block_size": int(
                gen_kwargs.get("block_length", gen_kwargs.get("inference_block_size", self.inference_block_size)) or 0
            ),
            "cfg_scale": float(gen_kwargs.get("cfg", gen_kwargs.get("cfg_scale", self.cfg_scale))),
            "remasking": str(gen_kwargs.get("remasking", self.remasking)),
        }

    def _log_generation_kwargs_once(
        self,
        *,
        task: str,
        split: str,
        raw_gen_kwargs: dict[str, Any],
        resolved_gen_kwargs: dict[str, Any],
        until: list[str],
    ) -> None:
        if self._logged_generation_kwargs or self.rank != 0:
            return
        eval_logger.info(
            "NanoMDM generation kwargs resolved: "
            f"task={task}, split={split}, prompt_style={self.prompt_style}, "
            f"raw={raw_gen_kwargs}, resolved={resolved_gen_kwargs}, until={until}, "
            f"stop_token_ids={get_prompt_contract_stop_token_ids(self.prompt_contract) if self.prompt_style == 'llada_header_mapped' else None}"
        )
        self._logged_generation_kwargs = True

    def generate_until(self, requests: List[Instance]) -> List[str]:
        responses: List[str] = []
        progress = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for req in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = req.args
            doc = self.task_dict[task][split][doc_id]
            visuals = self._flatten_visuals(doc_to_visual(doc))
            call_gen_kwargs = dict(gen_kwargs or {})
            until = self._normalize_until(call_gen_kwargs)
            resolved_gen_kwargs = self._resolve_generation_kwargs(call_gen_kwargs)
            self._log_generation_kwargs_once(
                task=task,
                split=split,
                raw_gen_kwargs=call_gen_kwargs,
                resolved_gen_kwargs=resolved_gen_kwargs,
                until=until,
            )
            result = generate_completion(
                self.bundle,
                self._build_segments(str(context), visuals),
                **resolved_gen_kwargs,
                stop_token_ids=(
                    get_prompt_contract_stop_token_ids(self.prompt_contract)
                    if self.prompt_style == "llada_header_mapped"
                    else None
                ),
            )
            result = self._apply_until(result, until)
            responses.append(result)
            self.cache_hook.add_partial("generate_until", req.args, result)
            progress.update(1)
        progress.close()
        return responses
