from diffusers import DiffusionPipeline
import numpy as np
from PIL import Image
from transformers import AutoProcessor
import torch
import argparse
import os
from blip3o.constants import *
from blip3o.conversation import conv_templates
from blip3o.model.builder import load_pretrained_model
from blip3o.utils import disable_torch_init
from blip3o.mm_utils import get_model_name_from_path
import math
from blip3o.conversation import conv_templates
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

import random
from pathlib import Path


def create_image_grid(images, rows, cols):
    """Creates a grid of images and returns a single PIL Image."""

    assert len(images) == rows * cols

    width, height = images[0].size
    grid_width = width * cols
    grid_height = height * rows

    grid_image = Image.new('RGB', (grid_width, grid_height))

    for i, image in enumerate(images):
        x = (i % cols) * width
        y = (i // cols) * height
        grid_image.paste(image, (x, y))

    return grid_image


def add_template(prompt):
   conv = conv_templates['qwen'].copy()
   conv.append_message(conv.roles[0], prompt[0])
   conv.append_message(conv.roles[1], None)
   prompt = conv.get_prompt()
   return [prompt]



def set_global_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _as_list(x):
    # 支持 list/tuple、逗号分隔字符串、单个字符串
    if isinstance(x, (list, tuple)):
        return list(x)
    if isinstance(x, str) and "," in x:
        return [s.strip() for s in x.split(",") if s.strip()]
    return [x] if x is not None and x != "" else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["t2i", "reconstruct", "edit", "autoencode"], required=True,
                        help="inference 类别：t2i(文本生成图像) / reconstruct(重建) / edit(编辑) / autoencode")
    parser.add_argument("--input",  nargs="+", help="一个或多个输入图像")
    parser.add_argument("--prompt", nargs="+", help="一个或多个提示词")
    parser.add_argument("--output", nargs="+", help="一个或多个输出路径")
    parser.add_argument("--num", type=int, default=4, help="t2i 生成张数（默认 4）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance_scale", type=float, default=3.0)
    # === 新增：模型相关参数 ===
    parser.add_argument("--model", required=True, help="多模态大模型权重路径（原 sys.argv[1]）")
    parser.add_argument("--pipeline",  default="pipeline_llava_gen",  help="pipeline 类型")
    parser.add_argument("--diffusion_path", default="/home/notebook/code/group/xuekaiwen/data/BLIP3o/BLIP3o-Model-8B/diffusion-decoder",
                        help="扩散解码器权重路径")
    parser.add_argument("--device", type=int, default=0, help="使用的 CUDA 设备编号")
    args = parser.parse_args()

    print("args.input:", args.input)

    # ====== 将你给的初始化代码并入 main()（仅做最小改动）======
    model_path = os.path.expanduser(args.model)
    # diffusion_path = model_path + "/diffusion-decoder"
    diffusion_path = os.path.expanduser(args.diffusion_path)

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    device_1 = args.device  # 原本写死为 0，这里接入命令行

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, multi_model, context_len = load_pretrained_model(model_path, None, model_name)

    pipe = DiffusionPipeline.from_pretrained(
        diffusion_path,
        custom_pipeline=args.pipeline,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        variant="bf16",
        multimodal_encoder=multi_model,
        tokenizer=tokenizer,
        safety_checker=None
    )

    device_str = f"cuda:{device_1}" if torch.cuda.is_available() else "cpu"
    if device_str.startswith("cuda"):
        pipe.vae.to(device_str)
        pipe.unet.to(device_str)

    # ====== 下方保持你原有推理逻辑（最小修改）======
    set_global_seed(seed=args.seed)
    gen_images = []

    if args.task == "t2i":
        if not args.prompt:
            raise ValueError("t2i 需要 --prompt")
        for _ in range(args.num):
            inputs = add_template(
                [f"Please generate image based on the following caption: {args.prompt}"]
            )
            gen = pipe(inputs, guidance_scale=args.guidance_scale)
            gen_images.append(gen.image)
        n = args.num
        rows = cols = int(math.ceil(n ** 0.5))
        grid_image = create_image_grid(gen_images, rows, cols)
        out = Path(args.output) if args.output else Path(f"{args.prompt[:100]}.png")
        grid_image.save(out, format="JPEG")

    elif args.task == "reconstruct":
        if not args.input:
            raise ValueError("reconstruct 需要 --input")
        inputs_list  = _as_list(args.input)
        outputs_list = _as_list(args.output)

        # 广播规则：单个 output 自动扩展到所有输入
        if len(outputs_list) == 1 and len(inputs_list) > 1:
            outputs_list = outputs_list * len(inputs_list)

        # 若未提供 outputs，则为每个输入按默认规则命名
        if len(outputs_list) == 0:
            outputs_list = ["" for _ in inputs_list]

        assert len(inputs_list) == len(outputs_list), \
            f"inputs/outputs 数量不一致：{len(inputs_list)}, {len(outputs_list)}"

        for img_path, out_path in zip(inputs_list, outputs_list):

            p = Path(img_path)
            out = Path(out_path) if out_path else p.with_name(p.stem + "_recon.png")
            inputs = add_template(["<image>\nPlease reconstruct the given image."])
            inputs.append(Image.open(str(p)))
            gen = pipe(inputs, guidance_scale=args.guidance_scale)
            gen.image.save(out, format="JPEG")

    elif args.task == "autoencode":
        if not args.input:
            raise ValueError("autoencode 需要 --input")
        inputs_list  = _as_list(args.input)
        outputs_list = _as_list(args.output)

        # 广播规则：单个 output 自动扩展到所有输入
        if len(outputs_list) == 1 and len(inputs_list) > 1:
            outputs_list = outputs_list * len(inputs_list)

        # 若未提供 outputs，则为每个输入按默认规则命名
        if len(outputs_list) == 0:
            outputs_list = ["" for _ in inputs_list]

        assert len(inputs_list) == len(outputs_list), \
            f"inputs/outputs 数量不一致：{len(inputs_list)}, {len(outputs_list)}"

        for img_path, out_path in zip(inputs_list, outputs_list):

            p = Path(img_path)
            out = Path(out_path) if out_path else p.with_name(p.stem + "_ae.png")
            inputs = []
            inputs.append(Image.open(str(p)))
            gen = pipe(inputs, guidance_scale=args.guidance_scale)
            gen.image.save(out, format="JPEG")

    elif args.task == "edit":
        if not args.input or not args.prompt:
            raise ValueError("edit 需要 --input 与 --prompt")
        inputs_list  = _as_list(args.input)
        prompts_list = _as_list(args.prompt)
        outputs_list = _as_list(args.output)

        # 广播规则：单个 prompt/output 自动扩展到所有输入
        if len(prompts_list) == 1 and len(inputs_list) > 1:
            prompts_list = prompts_list * len(inputs_list)
        if len(outputs_list) == 1 and len(inputs_list) > 1:
            outputs_list = outputs_list * len(inputs_list)

        # 若未提供 outputs，则为每个输入按默认规则命名
        if len(outputs_list) == 0:
            outputs_list = ["" for _ in inputs_list]

        assert len(inputs_list) == len(prompts_list) == len(outputs_list), \
            f"inputs/prompts/outputs 数量不一致：{len(inputs_list)}, {len(prompts_list)}, {len(outputs_list)}"

        for img_path, prompt, out_path in zip(inputs_list, prompts_list, outputs_list):
            p = Path(img_path)
            out = Path(out_path) if out_path else p.with_name(p.stem + "_i2i.png")

            inputs = add_template([f"<image>\n{prompt}"])
            inputs.append(Image.open(str(p)))

            gen = pipe(inputs, guidance_scale=args.guidance_scale)

            # 逐个保存（无需构网格）
            gen.image.save(out, format="JPEG")


if __name__ == "__main__":
    main()


