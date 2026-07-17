import os
import io
import copy
from dataclasses import dataclass, field
import json, ast, itertools
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import time
import torch, gc
import glob
import transformers
import tokenizers
import random
from blip3o.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_IDX_LLADA, DEFAULT_IM_START_TOKEN_LLADA, DEFAULT_IM_END_TOKEN_LLADA, DEFAULT_IM_START_TOKEN_IDX_LLADA
from torch.utils.data import Dataset
from blip3o.train.blip3o_trainer import blip3oTrainer
from blip3o import conversation as conversation_lib
from blip3o.model import *
from blip3o.mm_utils import tokenizer_image_token
from PIL import Image, ImageFile, ImageDraw, ImageFont
from datasets import load_dataset, load_from_disk, concatenate_datasets, Features, Value
from datasets import Image as HFImage
from datasets.features import Sequence as HFSeq, Image as HFFeatureImage
from pathlib import Path
from datetime import timedelta
import re
from datasets.utils.logging import set_verbosity_info
from transformers import logging as tf_logging
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor
from transformers import TrainerCallback, TrainingArguments
from transformers.trainer_callback import PrinterCallback, ProgressCallback
from transformers.trainer_utils import IntervalStrategy
from datetime import datetime
import torch.distributed as dist
try:
    from zoneinfo import ZoneInfo  # Py>=3.9
except Exception:
    ZoneInfo = None
from io import BytesIO
from PIL import Image as PILImage, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
transform_und_images = T.Compose([T.Resize(448, interpolation=InterpolationMode.BICUBIC, antialias=True), T.CenterCrop(448)])

set_verbosity_info()
tf_logging.set_verbosity_info()

local_rank = None
os.environ["ACCELERATE_TIMEOUT"] = "21600"    # Accelerate 初始化 PG 的超时
os.environ["TORCHRUN_TIMEOUT"] = "21600"      # 一些环境会读取该值作为兜底（安全冗余）
os.environ["HF_DATASETS_DISABLE_MULTIPROCESSING"] = "1"

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def robust_load_image(img_field):
    """
    支持几种输入形态：
      - HF Image(decode=False) 给的 dict：{'bytes': b'...'} 或 {'path': '...'}
      - bytes/bytearray
      - str 路径
      - 已经是 PIL.Image.Image（会直接转 RGB）
    返回：PIL.Image (RGB)
    """
    def _open_pil_bytes(b: bytes):
        im = PILImage.open(BytesIO(b))
        return im.convert("RGB")

    # 已是 PIL 图像
    if hasattr(img_field, "convert") and hasattr(img_field, "size"):
        return img_field.convert("RGB")

    # HF dict
    if isinstance(img_field, dict):
        if img_field.get("bytes") is not None:
            b = img_field["bytes"]
            try:
                return _open_pil_bytes(b)
            except UnidentifiedImageError as e:
                raise UnidentifiedImageError(f"PIL failed: {e}")
        if img_field.get("path"):
            return PILImage.open(img_field["path"]).convert("RGB")

    # 原始字节
    if isinstance(img_field, (bytes, bytearray)):
        return _open_pil_bytes(bytes(img_field))

    # 文件路径
    if isinstance(img_field, str):
        return PILImage.open(img_field).convert("RGB")

    raise UnidentifiedImageError(f"Unknown image field type: {type(img_field)}")

from packaging import version

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse("0.14")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=True)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    gen_vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_gen_mlp_adapter: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    gen_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    n_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    n_und_query: Optional[int] = field(default=729)  # clip 576, siglip 729
    gen_pooling: Optional[str] = field(default="all")  # options are: pool2d_3, pool2d_9, seq_3, seq_9, seq_27
    add_faster_video: Optional[bool] = field(default=False)


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    journeyDB_folder: Optional[str] = field(default=None)
    irecon_folder: Optional[str] = field(default=None)
    shortcaption_image_folder: Optional[str] = field(default=None)
    video_icedit_folder: Optional[str] = field(default=None)
    video_icgen_folder: Optional[str] = field(default=None)
    data_type: Optional[str] = field(default="mix")
    image_aspect_ratio: str = "square"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    logging_strategy: IntervalStrategy = field(default=IntervalStrategy.STEPS)
    logging_steps: int = field(default=10)
    logging_first_step: bool = field(default=True)
    report_to: Optional[List[str]] = field(default_factory=list)  # 别用 [] 作为可变默认
    disable_tqdm: bool = field(default=True)

    ddp_timeout: int = 21600

    # 关键：在这里统一“更稳”的分布式默认
    def __post_init__(self):
        super().__post_init__()
        # 这些字段来自父类 transformers.TrainingArguments，这里只改默认值/下限，不重复声明字段
        # if self.ddp_bucket_cap_mb is None:
        #     self.ddp_bucket_cap_mb = 12                # 收小桶，避免一次超大 allreduce
        # if self.ddp_broadcast_buffers is None:
        #     self.ddp_broadcast_buffers = False         # 少同步 buffers，降通信噪声
        # self.dataloader_drop_last = True               # 保证各 rank 步数一致
        if self.ddp_timeout is None or self.ddp_timeout < 21600:
            self.ddp_timeout = 21600                    # 进程组超时（秒）

class StdoutLogCallback(TrainerCallback):
    def __init__(self, tz: str | None = None,
                 datefmt: str = "%Y-%m-%d %H:%M:%S %Z%z"):
        """
        tz: 例如 'Asia/Taipei'；None 则用系统本地时区
        datefmt: 时间格式，默认带时区名与偏移
        """
        self.tz = tz
        self.datefmt = datefmt

    def _now_str(self) -> str:
        if self.tz and ZoneInfo is not None:
            try:
                return datetime.now(ZoneInfo(self.tz)).strftime(self.datefmt)
            except Exception:
                pass
        # 回退到系统本地时区
        return datetime.now().astimezone().strftime(self.datefmt)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or not logs:
            return
        ts    = self._now_str()
        step  = state.global_step
        loss  = logs.get("loss")
        lr    = logs.get("learning_rate")
        epoch = logs.get("epoch")

        parts = [f"[{ts}] [train] step={step}"]
        if epoch is not None: parts.append(f"epoch={epoch:.2f}" if isinstance(epoch, float) else f"epoch={epoch}")
        if loss  is not None: parts.append(f"loss={float(loss):.6f}")
        if lr    is not None: parts.append(f"lr={float(lr):.6g}")
        print("  ".join(parts), flush=True)
def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_vision_tower_state_maybe_zero_3(named_params, keys_to_match=[""]):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def safe_stub(text: str, max_len=64):
    """把说明文字清洗为文件名安全片段"""
    text = (text or "").strip().replace("\n", " ")
    text = re.sub(r"\s+", " ", text)       # 合并空白
    text = text[:max_len]                   # 截断
    text = re.sub(r'[^A-Za-z0-9._-]+', '_', text)  # 非安全字符替换为下划线
    return text


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, vision_tower: str):
    """Collects the state dict and dump to disk."""

    # if getattr(trainer.args, "tune_vision_model", False):

    if trainer.deepspeed:
        torch.cuda.synchronize()
    

    # Only save Adapter
    keys_to_match = ["mm_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "mm_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"mm_projector.bin"))

    keys_to_match = ["gen_projector"]
    if getattr(trainer.args, "use_im_start_end", False):
        keys_to_match.extend(["embed_tokens", "embed_in"])

    weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
    trainer.model.config.save_pretrained(output_dir)

    current_folder = output_dir.split("/")[-1]
    parent_folder = os.path.dirname(output_dir)
    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if current_folder.startswith("checkpoint-"):
            mm_projector_folder = os.path.join(parent_folder, "gen_projector")
            os.makedirs(mm_projector_folder, exist_ok=True)
            torch.save(
                weight_to_save,
                os.path.join(mm_projector_folder, f"{current_folder}.bin"),
            )
        else:
            torch.save(weight_to_save, os.path.join(output_dir, f"gen_projector.bin"))

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):


    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    for tok in tokenizer.all_special_tokens:
        print(f"{tokenizer.convert_tokens_to_ids(tok)}\t{tok!r}")

    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation



def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    gen_placeholder = ""

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            if sentence["from"] == "human" and "<image>" in sentence["value"]:
                num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
                if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
                    sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                    sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                    sentence["value"] = sentence["value"].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN_LLADA + replace_token + DEFAULT_IM_END_TOKEN_LLADA
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

                # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
                sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")
            elif sentence["from"] == "gpt" and "<image>" in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, gen_placeholder).strip()

    return sources





def preprocess_qwen(sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.") -> Dict:
    roles = {"human": "user", "gpt": "assistant"}

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            encode_id = tokenizer.apply_chat_template(conv)
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"

        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )




def preprocess_llama3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
        

                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )

def preprocess_llada(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    tokenizer = copy.deepcopy(tokenizer)
    # When there is actually an image, we add the image tokens as a special token
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    rank0_print("image_token_index: ", image_token_index)
    bos_token_id = tokenizer.convert_tokens_to_ids("<|startoftext|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = ["<|startoftext|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "\n\n"]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]
    # Reset LLaDA chat templates so that it won't include assistant message every time we apply
    chat_template = "{% for message in messages %}{{'<|startoftext|>' + '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role =  roles.get(role, role)
            
            conv = [{"role" : role, "content" : content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id
                    
        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        # assert DEFAULT_IMAGE_TOKEN in source[0]['value'] or DEFAULT_IMAGE_TOKEN in source[1]['value']
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.version == "llama3":
        return preprocess_llama3(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "qwen":
        return preprocess_qwen(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "llava_llada":
        return preprocess_llada(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def count_inputs_fixed(ex):
    names = ["image", "image1", "image2"]
    return sum(ex.get(n) is not None for n in names)

def parse_json_field(j):
    """把 sources['json'] 统一转成 dict；支持 dict/str/bytes/file-like。
       先按 JSON 解析，失败再 literal_eval。"""
    if j is None:
        return {}
    # str: 先 JSON，再 literal_eval 兜底
    if isinstance(j, str):
        s = j.lstrip("\ufeff").strip()  # 去 BOM/空白
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                # 给调试用的上下文片段（可选）
                # print("Bad json snippet:", repr(s[:200]))
                return {}
    # file-like
    if hasattr(j, "read"):
        try:
            return json.load(j)
        except Exception:
            j = j.read()
    # bytes -> str
    if isinstance(j, (bytes, bytearray, memoryview)):
        j = bytes(j).decode("utf-8", errors="ignore")
    # 已是 dict
    if isinstance(j, dict):
        return j
    # 其他类型，尽量转成 str 再试
    try:
        return json.loads(str(j))
    except Exception:
        try:
            obj = ast.literal_eval(str(j))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

class LazySupervisedMixDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
    ):
        super(LazySupervisedMixDataset, self).__init__()

        self.data_args = data_args
        list_data_dict = []


        # load journeyDB_T2I data with json file 
        # train_dataset = load_dataset("json", data_files='/fsx/sfr/data/jiuhai/hub/datasets--JourneyDB--JourneyDB/snapshots/e191aa61ca37e5e4418707ade4df5deb5c6d5d8f/data/train/train_caption_only.jsonl', split="train", num_proc=64)
        # if args.journeyDB_folder is not None:
        #     train_dataset = load_dataset("json", data_files=os.path.join(args.journeyDB_folder, "data/train/train_caption_only.jsonl"), split="train", num_proc=64)
        #     train_dataset = train_dataset.add_column('type', len(train_dataset) * ['journeyDB_T2I'])
        #     train_dataset = train_dataset.add_column('image', len(train_dataset) * [None])
        #     train_dataset = train_dataset.rename_column("caption", "txt")
        #     train_dataset = train_dataset.rename_column("img_path", "image_path")
        #     train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
        #         ["txt", "image", "type", "image_path"])])
        #     print(f"finish loading journeyDB {len(train_dataset)}")
        
        def build_ds(data_files, cache_dir=None):
            kwargs = dict(path="webdataset", data_files=data_files, split="train", cache_dir=cache_dir)

            # 判定 rank0（优先用已初始化的 dist.get_rank()，否则退回环境变量）
            if int(os.environ.get("RANK", "0")) == 0:
                _ = load_dataset(**kwargs)  # 仅 rank0 预构建/写缓存

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                dist.barrier()

            ds = load_dataset(**kwargs)
            return ds
        

        ###################################### text to image ####################################### 
        if self.data_args.image_folder is not None:
            data_files = sorted(glob.glob(os.path.join(self.data_args.image_folder, "*.tar")))
            if "BLIP3o-60k" in self.data_args.image_folder:
                train_dataset = build_ds(data_files)
                train_dataset = train_dataset.rename_column("jpg", "image")
                train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
                train_dataset = train_dataset.rename_column('__url__', 'image_path')
                # train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
                #     ["image", "txt", "type", "image_path"])])
            else:
                train_dataset = build_ds(data_files)
                train_dataset = train_dataset.rename_column("jpg", "image")
                train_dataset = train_dataset.add_column('type', len(train_dataset) * ['T2I'])
                train_dataset = train_dataset.add_column('image_path', len(train_dataset) * [None])
                # train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
                #     ["image", "txt", "type", "image_path"])])
            print(f"finish loading image from {self.data_args.image_folder}, number of images:{len(train_dataset)}")
            list_data_dict.append(train_dataset)

        ###################################### image reconstruction ####################################### 
        if self.data_args.irecon_folder is not None:
            data_files = sorted(glob.glob(os.path.join(self.data_args.irecon_folder, "*.tar")))
            train_dataset = build_ds(data_files, num_proc=128)
            print("train_dataset: ", train_dataset)
            train_dataset = train_dataset.add_column('type', len(train_dataset) * ['I2I'])
            train_dataset = train_dataset.rename_column("__key__", "id")
            train_dataset = train_dataset.rename_column('__url__', 'image_path')
            train_dataset = train_dataset.rename_column("jpg", "image")
            print("train_dataset: ", train_dataset)
            print(f"finish loading image from {self.data_args.irecon_folder}, number of images:{len(train_dataset)}")
            
            # ---- 在数据集构造完成后执行一次 ----
            def _disable_hf_decode(ds):
                feat = ds.features.get("image", None)
                try:
                    if isinstance(feat, HFSeq) and isinstance(feat.feature, HFFeatureImage):
                        ds = ds.cast_column("image", HFSeq(HFImage(decode=False)))
                    elif isinstance(feat, HFFeatureImage):
                        ds = ds.cast_column("image", HFImage(decode=False))
                except Exception as e:
                    print(f"[WARN] cast_column('image', decode=False) 失败：{e}")
                return ds
            
            train_dataset = _disable_hf_decode(train_dataset)

            list_data_dict.append(train_dataset)
        
        ###################################### text and image to image (video in context edit)  ####################################### 
        if self.data_args.video_icedit_folder is not None:

            data_files = sorted(glob.glob(os.path.join(self.data_args.video_icedit_folder, "*.tar")))
            train_dataset = build_ds(data_files, num_proc=128)
            train_dataset = train_dataset.add_column('type', len(train_dataset) * ['TI2I'])
            train_dataset = train_dataset.rename_column("__key__", "id")
            train_dataset = train_dataset.rename_column("in0.png", "image")
            train_dataset = train_dataset.rename_column("out.png", "output_image")
            train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
                ['id', 'type', 'image', 'output_image', 'json'])])
                
            list_data_dict.append(train_dataset)

        ###################################### text and images to image (video in context generation)  ####################################### 
        if self.data_args.video_icgen_folder is not None:
            data_files = sorted(glob.glob(os.path.join(self.data_args.video_icgen_folder, "*.tar")))

            features = Features({
                "__key__": str,
                "input_images.000.png": HFImage(),
                "input_images.001.png": HFImage(),
                "input_images.002.png": HFImage(),
                "output_image.jpg": HFImage(),
                "json": Value("string"),  
                "__key__": Value("string"),  
            })
            train_dataset = load_dataset("webdataset", data_files=data_files, split="train", features=features, num_proc=128)
            # TODO modify the dict format
            train_dataset = train_dataset.add_column('type', len(train_dataset) * ['TII2I'])
            train_dataset = train_dataset.rename_column("__key__", "id")
            train_dataset = train_dataset.rename_column("input_images.000.png", "image")
            train_dataset = train_dataset.rename_column("input_images.001.png", "image1")
            train_dataset = train_dataset.rename_column("input_images.002.png", "image2")
            train_dataset = train_dataset.rename_column("output_image.jpg", "output_image")
            train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if not col in (
                ['id', 'type', 'image', "image1", "image2", 'output_image', 'json'])])
                
            list_data_dict.append(train_dataset)

            

        if len(list_data_dict) > 1:
            list_data_dict = concatenate_datasets(list_data_dict)
        else:
            list_data_dict = list_data_dict[0]
        list_data_dict = list_data_dict.shuffle(seed=42)

        rank0_print(f"Totoal number of training instance: {len(list_data_dict)}")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list



    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        while True:
            sources = self.list_data_dict[i]
            json_info = parse_json_field(sources.get("json"))
            if sources["type"] == "T2I" or sources["type"] == "journeyDB_T2I":
                sources["conversations"] = [
                    {"from": "human", "value": f"Please generate image based on the following caption: {sources['txt']}"},
                    {"from": "gpt", "value": "<image>"},
                ]
            elif sources["type"] == "I2I" or sources["type"] == "journeyDB_I2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\nPlease reconstruct the given image.",
                    },
                    {"from": "gpt", "value": ""},
                ]
            elif sources["type"] == "TI2I":
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": f"<image>\n{json_info['instruction']}",
                    },
                    {"from": "gpt", "value": "<image>"},
                ]
            elif sources["type"] == "TII2I":
                
                sources["conversations"] = [
                    {
                        "from": "human",
                        "value": "<image>" * int(count_inputs_fixed(sources)) + f"\n{json_info['instruction']}",
                    },
                    {"from": "gpt", "value": "<image>"},
                ]
            else:
                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")

            if "image" in sources:
                def _try_load_font(font_size: int):
                    """尽量加载可显示中英文的字体；找不到就退回 PIL 默认字体。"""
                    candidates = [
                        "/home/notebook/code/group/xuekaiwen/.cache/geneval_panel/DejaVuSans.ttf",
                    ]
                    for p in candidates:
                        try:
                            if os.path.exists(p):
                                return ImageFont.truetype(p, font_size)
                        except Exception:
                            pass
                    return ImageFont.load_default()

                def _render_caption_below(pil_img: Image.Image, caption: str,
                                        font_size: int = 20, padding: int = 16,
                                        text_color=(0, 0, 0), bg_color=(255, 255, 255)) -> Image.Image:
                    """在图片下方加文字区并渲染 caption，返回新图。"""
                    if not caption:
                        return pil_img

                    font = _try_load_font(font_size)
                    W = pil_img.width
                    # 先按像素宽度做逐字符/逐词换行（兼容中英文）
                    draw = ImageDraw.Draw(pil_img)
                    max_w = W - 2 * padding

                    def wrap_by_pixels(text):
                        lines, cur = [], ""
                        for ch in text:
                            test = cur + ch
                            w = draw.textlength(test, font=font)
                            if w <= max_w or cur == "":
                                cur = test
                            else:
                                lines.append(cur)
                                cur = ch
                        if cur:
                            lines.append(cur)
                        return lines

                    lines = []
                    for seg in caption.split("\n"):
                        lines.extend(wrap_by_pixels(seg))

                    # 计算文字区高度
                    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (font.size, 0)
                    line_h = ascent + descent + 2  # 行距
                    text_h = line_h * len(lines)
                    H_new = pil_img.height + text_h + 2 * padding

                    # 生成新画布并粘贴原图
                    out = Image.new("RGB", (W, H_new), color=bg_color)
                    out.paste(pil_img, (0, 0))
                    draw = ImageDraw.Draw(out)

                    # 文字起点
                    y = pil_img.height + padding
                    x = padding
                    for line in lines:
                        draw.text((x, y), line, fill=text_color, font=font)
                        y += line_h

                    return out

                def _hstack_align_height(pils, gap, bg):
                    """把多张 PIL 图按高度对齐后横向拼接成一张，返回拼好的 PIL。"""
                    H = max(im.height for im in pils)
                    resized = []
                    for im in pils:
                        if im.height != H:
                            im = im.resize((int(im.width * H / im.height), H), Image.BICUBIC)
                        resized.append(im)
                    W = sum(im.width for im in resized) + gap * (len(resized) - 1)
                    canvas = Image.new("RGB", (W, H), bg)
                    x = 0
                    for j, im in enumerate(resized):
                        canvas.paste(im, (x, 0))
                        x += im.width + (gap if j < len(resized) - 1 else 0)
                    return canvas

                def _prepare_save_images(raw_images, processor, image_aspect_ratio):
                    """生成用于落盘的 PIL 列表：当 aspect_ratio='pad' 时，做与 img_process 一致的方形填充。"""
                    out = []
                    if image_aspect_ratio == "pad":
                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result
                        bg = tuple(int(x * 255) for x in processor.image_mean)
                        for img in raw_images:
                            out.append(expand2square(img, bg).convert("RGB"))
                    else:
                        for img in raw_images:
                            out.append(img.convert("RGB"))
                    return out

                def img_process(images, processor, image_aspect_ratio):
                    if image_aspect_ratio == "pad":

                        def expand2square(pil_img, background_color):
                            width, height = pil_img.size
                            if width == height:
                                return pil_img
                            elif width > height:
                                result = Image.new(pil_img.mode, (width, width), background_color)
                                result.paste(pil_img, (0, (width - height) // 2))
                                return result
                            else:
                                result = Image.new(pil_img.mode, (height, height), background_color)
                                result.paste(pil_img, ((height - width) // 2, 0))
                                return result

                        images = [expand2square(img, tuple(int(x * 255) for x in processor.image_mean)) for img in images]
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    else:
                        images = processor.preprocess(images, return_tensors="pt")["pixel_values"]
                    return images

                if sources["type"] == "T2I" or sources["type"] == "I2I":
                    image_files = self.list_data_dict[i]["image"]
                    output_image_files = []
                elif sources["type"] == "TI2I":
                    image_files = self.list_data_dict[i]["image"]
                    output_image_files = self.list_data_dict[i]["output_image"]
                elif sources["type"] == "TII2I":
                    if self.list_data_dict[i]["image2"] is not None:
                        image_files = [self.list_data_dict[i]["image"], self.list_data_dict[i]["image1"], self.list_data_dict[i]["image2"]]
                    elif self.list_data_dict[i]["image1"] is not None:
                        image_files = [self.list_data_dict[i]["image"], self.list_data_dict[i]["image1"]]
                    else:
                        image_files = self.list_data_dict[i]["image"]
                    output_image_files = self.list_data_dict[i]["output_image"]
                else:
                    image_files = self.list_data_dict[i]["image_path"]
                    output_image_files = []

                if not isinstance(image_files, list):
                    image_files = [image_files]
                if not isinstance(output_image_files, list):
                    output_image_files = [output_image_files]
                    
                images = []
                output_images = []

                def read_bin_as_bytesio(bin_file_path):
                    with open(bin_file_path, "rb") as f:
                        return io.BytesIO(f.read())

                for img in image_files:
                    try:
                        if sources["type"] == "T2I" or sources["type"] == "TI2I" or sources["type"] == "TII2I":
                            img = img.convert("RGB")
                        elif sources["type"] == "I2I":
                            try:
                                img = robust_load_image(img)   # 一定返回 RGB（或抛异常）
                                # 如需额外保障，可再转一次（幂等）
                                img = img.convert("RGB")
                            except Exception as e:
                                # 仅跳过这张，不让整个样本立刻报错
                                print(f"[WARN] sample idx={i}, image failed to load: {e}")

                        elif sources["type"] == "journeyDB_T2I" or sources["type"] == "journeyDB_I2I":
                            if sources["type"] == "journeyDB_T2I" or sources["type"] == "journeyDB_I2I":
                                image_path = os.path.join(args.journeyDB_folder, "data", "train", "imgs", img)
                                # image_path = os.path.join('/fsx/sfr/data/jiuhai/hub/datasets--JourneyDB--JourneyDB/snapshots/e191aa61ca37e5e4418707ade4df5deb5c6d5d8f/data/train/imgs', img)
                            else:
                                raise ValueError("Unknown source type. Please check the 'type' in 'sources'.")
                            img = Image.open(image_path).convert("RGB")
                        images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        images = None
                        break  # Skip to the next image if there's an error

                for img in output_image_files:
                    try:
                        if sources["type"] == "TI2I" or sources["type"] == "TII2I":
                            img = img.convert("RGB")
                        output_images.append(img)
                    except Exception as e:
                        print(f"Error opening image {img}: {e}")
                        output_images = None
                        break  # Skip to the next image if there's an error

                if not images is None:
                    try:
                        # # --- DEBUG: 输入侧概览 ---
                        # try:
                        #     print(f"[debug][img] index={i}  images_in={len(images)}  aspect_ratio='{self.data_args.image_aspect_ratio}'")
                        #     for k, im in enumerate(images[:8]):  # 避免刷屏，只看前 8 张
                        #         mode = getattr(im, "mode", "?")
                        #         size = getattr(im, "size", "?")
                        #         print(f"  [debug][img] in[{k}] mode={mode} size={size}")
                        # except Exception as _:
                        #     pass
                        # --- 调用原处理函数 ---
                        temp = img_process(
                            images,
                            self.data_args.gen_image_processor,
                            self.data_args.image_aspect_ratio,
                        )
                        # # --- DEBUG: 输出侧概览（tensor/list 的形状/范围） ---
                        # try:
                        #     if isinstance(temp, torch.Tensor):
                        #         t = temp
                        #         tmin = float(t.min()) if t.numel() > 0 else float("nan")
                        #         tmax = float(t.max()) if t.numel() > 0 else float("nan")
                        #         tmean = float(t.mean()) if t.numel() > 0 else float("nan")
                        #         print(f"[debug][img] pixel_values: shape={tuple(t.shape)} dtype={t.dtype} "
                        #             f"min={tmin:.5f} max={tmax:.5f} mean={tmean:.5f}")
                        #     elif isinstance(temp, list) and len(temp) > 0 and isinstance(temp[0], torch.Tensor):
                        #         shapes = [tuple(x.shape) for x in temp[:8]]
                        #         print(f"[debug][img] pixel_values(list): n={len(temp)} head_shapes={shapes}")
                        #     else:
                        #         print(f"[debug][img] pixel_values type={type(temp)}")
                        # except Exception as _:
                        #     pass
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        images = None

                if not output_images is None:
                    try:
                        # # --- DEBUG: 输入侧概览 ---
                        # try:
                        #     print(f"[debug][img] index={i}  images_in={len(images)}  aspect_ratio='{self.data_args.image_aspect_ratio}'")
                        #     for k, im in enumerate(images[:8]):  # 避免刷屏，只看前 8 张
                        #         mode = getattr(im, "mode", "?")
                        #         size = getattr(im, "size", "?")
                        #         print(f"  [debug][img] in[{k}] mode={mode} size={size}")
                        # except Exception as _:
                        #     pass
                        # --- 调用原处理函数 ---
                        temp = img_process(
                            output_images,
                            self.data_args.gen_image_processor,
                            self.data_args.image_aspect_ratio,
                        )
                        # # --- DEBUG: 输出侧概览（tensor/list 的形状/范围） ---
                        # try:
                        #     if isinstance(temp, torch.Tensor):
                        #         t = temp
                        #         tmin = float(t.min()) if t.numel() > 0 else float("nan")
                        #         tmax = float(t.max()) if t.numel() > 0 else float("nan")
                        #         tmean = float(t.mean()) if t.numel() > 0 else float("nan")
                        #         print(f"[debug][img] pixel_values: shape={tuple(t.shape)} dtype={t.dtype} "
                        #             f"min={tmin:.5f} max={tmax:.5f} mean={tmean:.5f}")
                        #     elif isinstance(temp, list) and len(temp) > 0 and isinstance(temp[0], torch.Tensor):
                        #         shapes = [tuple(x.shape) for x in temp[:8]]
                        #         print(f"[debug][img] pixel_values(list): n={len(temp)} head_shapes={shapes}")
                        #     else:
                        #         print(f"[debug][img] pixel_values type={type(temp)}")
                        # except Exception as _:
                        #     pass
                    except Exception as e:
                        print(f"Error wrong number of channels: {e}")
                        output_images = None
                    

                # If no valid images were found, randomly pick another item
                if images is None:
                    print(sources)
                    print(f"warning false image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue

                if output_images is None and (sources["type"] == "TI2I" or sources["type"] == "TI2I"):
                    print(sources)
                    print(f"warning false output image!!!!!!")
                    i = random.randint(0, len(self.list_data_dict) - 1)
                    continue

                sources = preprocess_multimodal(copy.deepcopy([sources["conversations"]]), self.data_args)
            else:
                sources = copy.deepcopy([sources["conversations"]])

            data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))

            if isinstance(i, int):
                data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

            # image exist in the data
            if "image" in self.list_data_dict[i]:
                if self.list_data_dict[i]["type"] == "T2I":
                    data_dict["gen_image"] = img_process(
                        images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )
                    # # === 新增：保存“最终用到的图片”到 data_test/*.jpg ===
                    # try:
                    #     save_dir = "data_test"
                    #     os.makedirs(save_dir, exist_ok=True)
                    #     save_imgs = _prepare_save_images(
                    #         images,
                    #         self.data_args.gen_image_processor,
                    #         self.data_args.image_aspect_ratio,
                    #     )
                    #     base_id = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else f"idx{i}"
                    #     caption = self.list_data_dict[i].get("txt", "")  # T2I/journeyDB_T2I 通常有 txt
                    #     image_path = self.list_data_dict[i]['image_path'] if self.list_data_dict[i]['image_path'] else ""
                    #     # 若想与 pad 匹配背景色：
                    #     bg = tuple(int(x * 255) for x in self.data_args.gen_image_processor.image_mean)
                    #     for k, pil in enumerate(save_imgs):
                    #         final = _render_caption_below(pil, caption, font_size=20, padding=16,
                    #                                     text_color=(0, 0, 0), bg_color=bg)
                    #         out_path = os.path.join(save_dir, f"src_{Path(image_path).name}_id_{base_id}_gen_{k}_cap_{safe_stub(caption, 80)} ... <trunc>.jpg")
                    #         final.save(out_path, format="JPEG")
                    # except Exception as e:
                    #     print(f"[debug][save] gen save failed: {e}")
                elif self.list_data_dict[i]["type"] == "I2T":


                    image_inputs = self.data_args.image_processor.preprocess(images, return_tensors="pt")

                    data_dict["und_image"] = image_inputs.pixel_values
                    data_dict["gen_image"] = img_process(
                        resized_images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )
                elif self.list_data_dict[i]["type"] == "TI2I" or self.list_data_dict[i]["type"] == "TII2I":

                    image_inputs = self.data_args.image_processor.preprocess(images, return_tensors="pt")

                    data_dict["und_image"] = image_inputs.pixel_values

                    resized_output_images = [transform_und_images(img) for img in output_images]
                    data_dict["gen_image"] = img_process(
                        resized_output_images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )
                    # === 新增：保存 und_image + gen_image + instruction 为一张图片 ===
                    if False:
                        try:
                            base_id = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else f"idx{i}"
                            instruction = json_info.get("instruction", "")
                            # 与 pad 背景色一致
                            bg = tuple(int(x * 255) for x in self.data_args.gen_image_processor.image_mean)
                            gap = 16  # 横向间距

                            env_dir = (os.environ.get("SAVE_DIR") or "data_test")
                            save_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(env_dir)))
                            os.makedirs(save_dir, exist_ok=True)

                            # 与示例一致的准备函数，得到可直接保存的 PIL 图像
                            und_pils = _prepare_save_images(
                                resized_images,
                                self.data_args.gen_image_processor,
                                self.data_args.image_aspect_ratio,
                            )
                            gen_pils = _prepare_save_images(
                                resized_output_images,
                                self.data_args.gen_image_processor,
                                self.data_args.image_aspect_ratio,
                            )

                            # --- 通用 k-und 对 1-gen 的分组逻辑 ---
                            if len(gen_pils) > 0 and len(und_pils) >= len(gen_pils) and (len(und_pils) % len(gen_pils) == 0):
                                und_per_gen = len(und_pils) // len(gen_pils)  # 任意 k>=1
                                for j, g in enumerate(gen_pils):
                                    und_group = und_pils[j * und_per_gen : (j + 1) * und_per_gen]

                                    # 先把该组多张 und 对齐高度并横向拼成一个块
                                    if len(und_group) == 1:
                                        u_block = und_group[0]
                                    else:
                                        u_block = _hstack_align_height(und_group, gap, bg)

                                    # 与 gen 对齐高度后再横向拼
                                    H = max(u_block.height, g.height)
                                    if u_block.height != H:
                                        u_block = u_block.resize((int(u_block.width * H / u_block.height), H), Image.BICUBIC)
                                    if g.height != H:
                                        g = g.resize((int(g.width * H / g.height), H), Image.BICUBIC)

                                    combo = Image.new("RGB", (u_block.width + gap + g.width, H), bg)
                                    combo.paste(u_block, (0, 0))
                                    combo.paste(g, (u_block.width + gap, 0))

                                    final = _render_caption_below(
                                        combo, instruction, font_size=20, padding=16,
                                        text_color=(0, 0, 0), bg_color=bg
                                    )
                                    out_path = os.path.join(save_dir, f"{base_id}_und{und_per_gen}_gen_{j}.jpg")
                                    final.save(out_path, format="JPEG")

                            else:
                                # 退化：无法整分的情况，按索引成对保存
                                n = min(len(und_pils), len(gen_pils))
                                for k in range(n):
                                    u, g = und_pils[k], gen_pils[k]

                                    # 对齐高度后横向拼接
                                    H = max(u.height, g.height)
                                    if u.height != H:
                                        u = u.resize((int(u.width * H / u.height), H), Image.BICUBIC)
                                    if g.height != H:
                                        g = g.resize((int(g.width * H / g.height), H), Image.BICUBIC)

                                    combo = Image.new("RGB", (u.width + gap + g.width, H), bg)
                                    combo.paste(u, (0, 0))
                                    combo.paste(g, (u.width + gap, 0))

                                    final = _render_caption_below(
                                        combo, instruction, font_size=20, padding=16,
                                        text_color=(0, 0, 0), bg_color=bg
                                    )
                                    out_path = os.path.join(save_dir, f"{base_id}_und_gen_{k}.jpg")
                                    final.save(out_path, format="JPEG")
                        except Exception as e:
                            print(f"[debug][save-cmp] failed: {e}")
                    # === 保存 und_image 为一张图、gen_image 为一张图，instruction 到 json（key=und图文件名）===
                    if False:
                        try:
                            base_id = self.list_data_dict[i].get("id", f"idx{i}")
                            instruction = json_info.get("instruction", "")

                            # 与 pad 背景一致
                            bg = tuple(int(x * 255) for x in self.data_args.gen_image_processor.image_mean)
                            gap = 16

                            env_dir = (os.environ.get("SAVE_DIR") or "data_test")
                            save_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(env_dir)))
                            os.makedirs(save_dir, exist_ok=True)

                            # 准备 PIL 图像
                            und_pils = _prepare_save_images(
                                resized_images,
                                self.data_args.gen_image_processor,
                                self.data_args.image_aspect_ratio,
                            )
                            gen_pils = _prepare_save_images(
                                resized_output_images,
                                self.data_args.gen_image_processor,
                                self.data_args.image_aspect_ratio,
                            )

                            # —— 合成“一张 und” ——
                            if len(und_pils) == 0:
                                und_img = Image.new("RGB", (512, 512), bg)
                            elif len(und_pils) == 1:
                                und_img = und_pils[0]
                            else:
                                # 按高度对齐后横向拼成一张
                                und_img = _hstack_align_height(und_pils, gap, bg)

                            # —— 合成“一张 gen” ——
                            if len(gen_pils) == 0:
                                gen_img = Image.new("RGB", (512, 512), bg)
                            elif len(gen_pils) == 1:
                                gen_img = gen_pils[0]
                            else:
                                gen_img = _hstack_align_height(gen_pils, gap, bg)

                            # 保存图片
                            und_path = os.path.join(save_dir, f"{base_id}_und.jpg")
                            gen_path = os.path.join(save_dir, f"{base_id}_gen.jpg")
                            und_img.save(und_path, format="JPEG")
                            gen_img.save(gen_path, format="JPEG")

                            # 保存 instruction 到 JSON：{ und文件名: instruction }
                            mapping = {os.path.basename(und_path): instruction}
                            json_path = os.path.join(save_dir, f"{base_id}.json")
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(mapping, f, ensure_ascii=False, indent=2)

                        except Exception as e:
                            print(f"[debug][save-simple] failed: {e}")
                elif self.list_data_dict[i]["type"] == "I2I":
                    image_inputs = self.data_args.image_processor.preprocess(images, return_tensors="pt")

                    data_dict["und_image"] = image_inputs.pixel_values

                    data_dict["gen_image"] = img_process(
                        resized_images,
                        self.data_args.gen_image_processor,
                        self.data_args.image_aspect_ratio,
                    )
                    # === 保存 und_image 为一张图、gen_image 为一张图，instruction 到 json（key=und图文件名）===
                    if False:
                        try:
                            base_id = self.list_data_dict[i].get("id", f"idx{i}")
                            instruction = json_info.get("instruction", "")

                            # 与 pad 背景一致
                            bg = tuple(int(x * 255) for x in self.data_args.gen_image_processor.image_mean)
                            gap = 16

                            env_dir = (os.environ.get("SAVE_DIR") or "data_test")
                            save_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(env_dir)))
                            os.makedirs(save_dir, exist_ok=True)

                            # 准备 PIL 图像
                            und_pils = _prepare_save_images(
                                resized_images,
                                self.data_args.gen_image_processor,
                                self.data_args.image_aspect_ratio,
                            )
                            # —— 合成“一张 und” ——
                            if len(und_pils) == 0:
                                und_img = Image.new("RGB", (512, 512), bg)
                            elif len(und_pils) == 1:
                                und_img = und_pils[0]
                            else:
                                # 按高度对齐后横向拼成一张
                                und_img = _hstack_align_height(und_pils, gap, bg)

                            # 保存图片
                            und_path = os.path.join(save_dir, f"{base_id}_und.jpg")
                            und_img.save(und_path, format="JPEG")

                            # 保存 instruction 到 JSON：{ und文件名: instruction }
                            mapping = {os.path.basename(und_path): instruction}
                            json_path = os.path.join(save_dir, f"{base_id}.json")
                            with open(json_path, "w", encoding="utf-8") as f:
                                json.dump(mapping, f, ensure_ascii=False, indent=2)

                        except Exception as e:
                            print(f"[debug][save-simple] failed: {e}")
                
            elif self.data_args.is_multimodal:
                crop_size = self.data_args.image_processor.crop_size
                data_dict["image"] = torch.zeros(3, crop_size["height"], crop_size["width"])

            data_dict["ids"] = self.list_data_dict[i]["id"] if "id" in self.list_data_dict[i] else "unk"
            return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "ids"))
        multi_input_ids = []
        multi_labels = []
        i_s_pos = []
        for input_id, label in zip(input_ids, labels):
            input_id = input_id[: self.tokenizer.model_max_length - 65]
            label = label[: self.tokenizer.model_max_length - 65]
            i_s_pos.append(input_id.shape[0]+1)
            # TODO check the IMAGE_TOKEN_IDX and IMG in llada tokenizer
            img_id = torch.full((65,), IMAGE_TOKEN_IDX_LLADA, dtype=input_id.dtype, device=input_id.device)
            img_id[0] = DEFAULT_IM_START_TOKEN_IDX_LLADA
            input_id = torch.cat([input_id, img_id])
            img_label = torch.full((65,), IMAGE_TOKEN_IDX_LLADA, dtype=label.dtype, device=label.device)
            img_label[0] = DEFAULT_IM_START_TOKEN_IDX_LLADA
            label = torch.cat([label, img_label])
            multi_input_ids.append(input_id)
            multi_labels.append(label)

        input_ids = multi_input_ids
        labels = multi_labels
        if self.tokenizer.pad_token_id is None:
            rank0_print("set self.tokenizer.pad_token_id = 0")
            self.tokenizer.pad_token_id = 0 # This gets the best result. Don't know why.

        input_ids = self.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = self.pad_sequence(labels, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            
        if input_ids.shape[1] > self.tokenizer.model_max_length:
            print(f"Warning input with length {input_ids.shape[1]} is longer than max length {self.tokenizer.model_max_length}")
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels
        )

        batch_gen_images = []
        batch_und_images = []

        for instance in instances:
            if "gen_image" in instance:
                batch_gen_images.append(instance["gen_image"])


        if len(batch_gen_images) > 0:
            if all(x is not None and y.shape == batch_gen_images[0][0].shape for x in batch_gen_images for y in x):
                batch["gen_image"] = torch.cat([images for images in batch_gen_images], dim=0)
            else:
                batch["gen_image"] = batch_gen_images
        else:
            batch["gen_image"] = None


        for instance in instances:
            if "und_image" in instance:
                batch_und_images.append(instance["und_image"].unsqueeze(0))  ## 1*3*384*384


        # print(f"batch_und_images {batch_und_images}")
        if len(batch_und_images) > 0:
            batch["und_image"] = torch.cat([images for images in batch_und_images], dim=0)
        else:
            batch["und_image"] = None

        batch["ids"] = ids

        batch["i_s_pos"] = i_s_pos

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:

    if data_args.data_type == "mix":
        train_dataset = LazySupervisedMixDataset(tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)
    else:
        raise ValueError("Unknown data type. Please check the Dataloader type.")

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def unlock_vit(training_args, model_args, vision_tower):
    for n, p in vision_tower.named_parameters():
        p.requires_grad = True


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(model_args, data_args, training_args)
    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )
        
    ## if there exists vision tower for image understanind, we will load LLaMA LLM, otherwise will load Qwen-VL
    if "llada" in model_args.model_name_or_path.lower():
        model = blip3oLlavaLLaDAModelLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    elif model_args.vision_tower is not None:
        model = blip3oLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )
    else:
        model = blip3oQwenForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args,
        )

    model.config.use_cache = False

    if model_args.freeze_backbone:
        for (n, p) in model.get_model().named_parameters():
            p.requires_grad = False
        if "llada" not in model_args.model_name_or_path.lower():
            for (n, p) in model.visual.named_parameters():
                p.requires_grad = False
        for (n, p) in model.lm_head.named_parameters():
            p.requires_grad = False
    
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            print("set tokenizer.pad_token = tokenizer.unk_token")
            tokenizer.pad_token = tokenizer.unk_token

        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(
                    pad_token="<pad>",
                    additional_special_tokens=["[IMG]", "[/IMG]", "<image>"],
                ),
                tokenizer=tokenizer,
                model=model,
            )
        elif not "<image>" in tokenizer.get_added_vocab():
            rank0_print("add [IMG], [/IMG], <image> to tokenizer")
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(additional_special_tokens=["[IMG]", "[/IMG]", "<image>"]),
                tokenizer=tokenizer,
                model=model,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    rank0_print(f"Using conversation format: {conversation_lib.default_conversation.version}")


    
    # if model_args.vision_tower is not None:
    model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

    ## generation vision tower
    gen_vision_tower = model.get_gen_vision_tower()
    gen_vision_tower.to(
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
        device=training_args.device,
    )
    gen_vision_tower.requires_grad_(False)

    vision_tower = model.get_vision_tower()
    vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
    vision_tower.requires_grad_(False)

    data_args.gen_image_processor = gen_vision_tower.image_processor
    data_args.image_processor = vision_tower.image_processor

    data_args.is_multimodal = True

    data_args.n_query = model_args.n_query
    data_args.n_und_query = model_args.n_und_query

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.tokenizer_padding_side = tokenizer.padding_side
    model.config.tokenizer_model_max_length = tokenizer.model_max_length

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    # Calculate total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.get_model().parameters())
    trainable_params = sum(p.numel() for p in model.get_model().parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")

    model.config.add_faster_video = model_args.add_faster_video
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    trainer = blip3oTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[StdoutLogCallback(tz="Asia/Taipei")],
        **data_module,
    )
    trainer.remove_callback(PrinterCallback)
    
    from tabulate import tabulate

    if trainer.is_world_process_zero():
        stat = []
        for i, (n, p) in enumerate(trainer.model.named_parameters()):
            stat.append([i, n, p.shape, p.requires_grad])
        print(tabulate(stat, headers=["idx", "name", "shape", "trainable"]))
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print("Checkpoint found, resuming training.")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir,
        vision_tower=model_args.vision_tower,
    )


if __name__ == "__main__":
    train()

