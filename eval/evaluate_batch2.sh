#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${ROOT_DIR}"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

LMMS_ROOT="${LMMS_ROOT:-${ROOT_DIR}/remote/BLIP3o/eval/lmms-eval}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_PATH="${OUTPUT_PATH:-${ROOT_DIR}/outputs/benchmark/batch2_evaluator_v2}"
GPU_IDS="${GPU_IDS:-0,1,2,3,4,5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
USE_ACCELERATE="${USE_ACCELERATE:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
LIMIT="${LIMIT:-}"
SLEEP_SECONDS="${SLEEP_SECONDS:-10}"
DRY_RUN="${DRY_RUN:-0}"

MODEL_PROFILE="${MODEL_PROFILE:-mdm}"
GEN_PROFILE="${GEN_PROFILE:-auto}"

TASK_NAMES="${TASK_NAMES:-vqav2_val,gqa,vizwiz_vqa_val,textvqa_val,scienceqa_img,mmbench_en_dev,seedbench,pope,mme,ok_vqa_val2014,coco_karpathy_test,textcaps_val}"

MDM_SHORT_TOKENS="${MDM_SHORT_TOKENS:-4}"
MDM_VQA_TOKENS="${MDM_VQA_TOKENS:-16}"
MDM_CAPTION_TOKENS="${MDM_CAPTION_TOKENS:-64}"
MDM_CFG="${MDM_CFG:-0}"
MDM_REMASKING="${MDM_REMASKING:-low_confidence}"

AR_SHORT_TOKENS="${AR_SHORT_TOKENS:-4}"
AR_VQA_TOKENS="${AR_VQA_TOKENS:-16}"
AR_CAPTION_TOKENS="${AR_CAPTION_TOKENS:-64}"

append_model_arg() {
  local item="$1"
  if [[ -z "${item}" ]]; then
    return
  fi
  if [[ -n "${MODEL_ARGS:-}" ]]; then
    MODEL_ARGS="${MODEL_ARGS},${item}"
  else
    MODEL_ARGS="${item}"
  fi
}

configure_model_args() {
  if [[ -n "${MODEL_ARGS:-}" ]]; then
    case "${MODEL_PROFILE}" in
      qwen3_vl|qwen3|qwen_baseline)
        MODEL_WRAPPER="${MODEL_WRAPPER:-qwen3_vl}"
        ;;
      custom)
        if [[ -z "${MODEL_WRAPPER:-}" ]]; then
          echo "[batch2] MODEL_PROFILE=custom with MODEL_ARGS still requires MODEL_WRAPPER." >&2
          exit 2
        fi
        ;;
      *)
        MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
        ;;
    esac
    return
  fi

  case "${MODEL_PROFILE}" in
    mdm|nanomdm_mdm)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/interleaved_eval_3b_latest_20260409_latest_ckpt_main512_ci/exports/dlm_step9695/hf_mupfix}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/difflm/training_checkpoints/dlm_3p35b_from_s4_trusted_center_20260404_012318/iter_0009695}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    mdm_sft_llada|mdm_step7275_llada|nanomdm_mdm_sft_llada)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/finetune/llava_next_finetune_v1/nanomdm_llava_next_finetune_3b_mdm_step7275_llada_header_mapped_2n2g_v1}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/difflm/training_checkpoints/dlm_3p35b_from_s4_trusted_center_20260404_012318/iter_0007275}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    mdm_sft_4n8g_step7275|nanomdm_mdm_sft_4n8g_step7275)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/finetune/llava_next_finetune_v1/nanomdm_llava_next_finetune_3b_mdm_step7275_llada_header_mapped_4n8g_gbs64_v1/checkpoint-11540}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/difflm/training_checkpoints/dlm_3p35b_from_s4_trusted_center_20260404_012318/iter_0007275}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    mdm_sft_4n8g_step9695|nanomdm_mdm_sft_4n8g_step9695)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/finetune/llava_next_finetune_v1/nanomdm_llava_next_finetune_3b_mdm_step9695_llada_header_mapped_4n8g_gbs64_v1/checkpoint-11540}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/difflm/training_checkpoints/dlm_3p35b_from_s4_trusted_center_20260404_012318/iter_0009695}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    mdm_sft_recap558k_step9695|nanomdm_mdm_sft_recap558k_step9695)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/finetune/llava_recap_558k_finetune_v1/nanomdm_llava_recap558k_finetune_3b_mdm_step9695_sft11540_llada_header_mapped_4n8g_gbs64_v1/checkpoint-8721}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/difflm/training_checkpoints/dlm_3p35b_from_s4_trusted_center_20260404_012318/iter_0009695}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    ar|nanomdm_ar)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/interleaved_eval_3b_latest_20260409_latest_ckpt_main512_ci/exports/ar_step9695/hf}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/ar/training_checkpoints/ar_3p35b_earlyfusion_baseline_20260404_013839/iter_0009695}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    ar_sft_4n8g_step9695|nanomdm_ar_sft_4n8g_step9695)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/finetune/llava_next_finetune_v1/nanomdm_llava_next_finetune_3b_ar_step9695_llada_header_mapped_4n8g_gbs64_v2/checkpoint-11540}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/ar/training_checkpoints/ar_3p35b_earlyfusion_baseline_20260404_013839/iter_0009695}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    ar_sft_recap558k_step9695|nanomdm_ar_sft_recap558k_step9695)
      MODEL_WRAPPER="${MODEL_WRAPPER:-nanomdm_interleaved}"
      PRETRAINED="${PRETRAINED:-${ROOT_DIR}/outputs/finetune/llava_recap_558k_finetune_v1/nanomdm_llava_recap558k_finetune_3b_ar_step9695_sft11540_llada_header_mapped_4n8g_gbs64_v1/checkpoint-8721}"
      TRAINING_CKPT_DIR="${TRAINING_CKPT_DIR:-${ROOT_DIR}/remote/MegaDLMs/checkpoints/cache/ar/training_checkpoints/ar_3p35b_earlyfusion_baseline_20260404_013839/iter_0009695}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "training_ckpt_dir=${TRAINING_CKPT_DIR}"
      append_model_arg "prompt_style=${PROMPT_STYLE:-auto}"
      append_model_arg "torch_dtype=${TORCH_DTYPE:-auto}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-flash_attention_2}"
      ;;
    qwen3_vl|qwen3|qwen_baseline)
      MODEL_WRAPPER="${MODEL_WRAPPER:-qwen3_vl}"
      PRETRAINED="${PRETRAINED:-/home/notebook/code/sharedGroup/rg-image-edit/lsy/weights/qwen3_vl_8b}"
      MODEL_ARGS=""
      append_model_arg "pretrained=${PRETRAINED}"
      append_model_arg "dtype=${DTYPE:-bfloat16}"
      append_model_arg "attn_implementation=${ATTN_IMPLEMENTATION:-sdpa}"
      append_model_arg "device_map=${DEVICE_MAP:-auto}"
      ;;
    custom)
      if [[ -z "${MODEL_WRAPPER:-}" || -z "${MODEL_ARGS:-}" ]]; then
        echo "[batch2] MODEL_PROFILE=custom requires MODEL_WRAPPER and MODEL_ARGS." >&2
        exit 2
      fi
      ;;
    *)
      echo "[batch2] Unsupported MODEL_PROFILE=${MODEL_PROFILE}. Use mdm, mdm_sft_llada, mdm_sft_4n8g_step7275, mdm_sft_4n8g_step9695, mdm_sft_recap558k_step9695, ar, ar_sft_4n8g_step9695, ar_sft_recap558k_step9695, qwen3_vl, or custom." >&2
      exit 2
      ;;
  esac
}

resolved_gen_profile() {
  if [[ "${GEN_PROFILE}" != "auto" ]]; then
    echo "${GEN_PROFILE}"
    return
  fi
  case "${MODEL_PROFILE}" in
    mdm|nanomdm_mdm|mdm_sft_llada|mdm_step7275_llada|nanomdm_mdm_sft_llada|mdm_sft_4n8g_step7275|nanomdm_mdm_sft_4n8g_step7275|mdm_sft_4n8g_step9695|nanomdm_mdm_sft_4n8g_step9695|mdm_sft_recap558k_step9695|nanomdm_mdm_sft_recap558k_step9695)
      echo "mdm"
      ;;
    *)
      echo "ar"
      ;;
  esac
}

mdm_gen_kwargs() {
  local task="$1"
  case "${task}" in
    mmbench_en_dev|seedbench|pope|mme|scienceqa_img|mmbench_en_dev_fast200|seedbench_fast200|mme_fast280)
      echo "max_new_tokens=${MDM_SHORT_TOKENS},sample_steps=${MDM_SHORT_TOKENS},inference_block_size=${MDM_SHORT_TOKENS},cfg=${MDM_CFG},remasking=${MDM_REMASKING},temperature=0"
      ;;
    vqav2_val|gqa|vizwiz_vqa_val|textvqa_val|ok_vqa_val2014)
      echo "max_new_tokens=${MDM_VQA_TOKENS},sample_steps=${MDM_VQA_TOKENS},inference_block_size=${MDM_VQA_TOKENS},cfg=${MDM_CFG},remasking=${MDM_REMASKING},temperature=0"
      ;;
    coco_karpathy_test|textcaps_val)
      echo "max_new_tokens=${MDM_CAPTION_TOKENS},sample_steps=${MDM_CAPTION_TOKENS},inference_block_size=${MDM_CAPTION_TOKENS},cfg=${MDM_CFG},remasking=${MDM_REMASKING},temperature=0,top_p=1.0,num_beams=1"
      ;;
    *)
      echo "max_new_tokens=${MDM_VQA_TOKENS},sample_steps=${MDM_VQA_TOKENS},inference_block_size=${MDM_VQA_TOKENS},cfg=${MDM_CFG},remasking=${MDM_REMASKING},temperature=0"
      ;;
  esac
}

ar_gen_kwargs() {
  local task="$1"
  case "${task}" in
    mmbench_en_dev|seedbench|pope|mme|scienceqa_img|mmbench_en_dev_fast200|seedbench_fast200|mme_fast280)
      echo "max_new_tokens=${AR_SHORT_TOKENS},temperature=0"
      ;;
    vqav2_val|gqa|vizwiz_vqa_val|textvqa_val|ok_vqa_val2014)
      echo "max_new_tokens=${AR_VQA_TOKENS},temperature=0"
      ;;
    coco_karpathy_test|textcaps_val)
      echo "max_new_tokens=${AR_CAPTION_TOKENS},temperature=0,top_p=1.0,num_beams=1"
      ;;
    *)
      echo "max_new_tokens=${AR_VQA_TOKENS},temperature=0"
      ;;
  esac
}

gen_kwargs_for_task() {
  local task="$1"
  case "$(resolved_gen_profile)" in
    mdm)
      mdm_gen_kwargs "${task}"
      ;;
    lladav|llada_v)
      echo "temperature=0,cfg=${MDM_CFG},remasking=${MDM_REMASKING},gen_length=${LLADAV_GEN_LENGTH:-2},block_length=${LLADAV_BLOCK_LENGTH:-2},gen_steps=${LLADAV_GEN_STEPS:-2},think_mode=no_think"
      ;;
    ar|qwen|qwen3_vl)
      ar_gen_kwargs "${task}"
      ;;
    none)
      echo ""
      ;;
    *)
      echo "[batch2] Unsupported GEN_PROFILE=${GEN_PROFILE}. Use auto, mdm, lladav, ar, qwen3_vl, or none." >&2
      exit 2
      ;;
  esac
}

task_label() {
  local task="$1"
  case "${task}" in
    vqav2_val) echo "vqav2" ;;
    vizwiz_vqa_val) echo "vizwiz" ;;
    textvqa_val) echo "textvqa" ;;
    scienceqa_img) echo "scienceqa_img" ;;
    mmbench_en_dev) echo "mmbench" ;;
    ok_vqa_val2014) echo "okvqa" ;;
    coco_karpathy_test) echo "coco" ;;
    textcaps_val) echo "textcaps" ;;
    *) printf "%s" "${task}" | tr -c '[:alnum:]_.=-' '_' ;;
  esac
}

build_eval_command() {
  local task="$1"
  local output_path="$2"
  local gen_kwargs="$3"
  local -n out_cmd_ref="$4"

  if [[ "${USE_ACCELERATE}" == "1" ]]; then
    out_cmd_ref=("${PYTHON_BIN}" -m accelerate.commands.launch --num_processes=1 -m lmms_eval)
  else
    out_cmd_ref=("${PYTHON_BIN}" -m lmms_eval)
  fi

  out_cmd_ref+=(
    --model "${MODEL_WRAPPER}"
    --model_args "${MODEL_ARGS}"
    --tasks "${task}"
    --batch_size "${BATCH_SIZE}"
    --process_with_media
    --output_path "${output_path}"
  )

  if [[ -n "${gen_kwargs}" ]]; then
    out_cmd_ref+=(--gen_kwargs "${gen_kwargs}")
  fi
  if [[ -n "${LIMIT}" ]]; then
    out_cmd_ref+=(--limit "${LIMIT}")
  fi
  if [[ "${LOG_SAMPLES}" == "1" ]]; then
    out_cmd_ref+=(--log_samples --log_samples_suffix "${task}")
  fi
}

launch_task() {
  local slot="$1"
  local gpu="${GPUS[$slot]}"
  local task="$2"
  local label
  local gen_kwargs
  local task_output_path
  local log_file
  local cmd=()

  label="$(task_label "${task}")"
  gen_kwargs="$(gen_kwargs_for_task "${task}")"
  task_output_path="${OUTPUT_PATH}/${MODEL_PROFILE}/${label}"
  log_file="${task_output_path}/${label}_gpu${gpu}.log"
  mkdir -p "${task_output_path}"

  build_eval_command "${task}" "${task_output_path}" "${gen_kwargs}" cmd

  {
    echo "Task: ${task}"
    echo "Task label: ${label}"
    echo "Model profile: ${MODEL_PROFILE}"
    echo "Generation profile: $(resolved_gen_profile)"
    echo "Model wrapper: ${MODEL_WRAPPER}"
    echo "Model args: ${MODEL_ARGS}"
    echo "Generation kwargs: ${gen_kwargs}"
    echo "Output path: ${task_output_path}"
    echo "GPU ID: ${gpu}"
    printf "Command:"
    printf " %q" "${cmd[@]}"
    printf "\n"
    echo "----------------------------------------"
  } > "${log_file}"

  echo "[batch2] start ${task} on GPU ${gpu}; log = ${log_file}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PYTHONPATH="${LMMS_ROOT}:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf "[batch2] dry-run command:"
      printf " %q" "${cmd[@]}"
      printf "\n"
      exit 0
    fi
    "${cmd[@]}"
  ) >> "${log_file}" 2>&1 &

  GPU_STATUS[$slot]=1
  GPU_PIDS[$slot]=$!
  GPU_TASKS[$slot]="${task}"
  echo "[batch2] PID ${GPU_PIDS[$slot]} running ${task} on GPU ${gpu}"
}

configure_model_args

IFS=',' read -ra TASKS <<< "${TASK_NAMES}"
IFS=',' read -ra GPUS <<< "${GPU_IDS}"

if [[ ${#TASKS[@]} -eq 0 || -z "${TASKS[0]}" ]]; then
  echo "[batch2] No tasks configured." >&2
  exit 2
fi
if [[ ${#GPUS[@]} -eq 0 || -z "${GPUS[0]}" ]]; then
  echo "[batch2] No GPUs configured." >&2
  exit 2
fi

mkdir -p "${OUTPUT_PATH}/${MODEL_PROFILE}"

declare -a GPU_STATUS
declare -a GPU_PIDS
declare -a GPU_TASKS

for slot in "${!GPUS[@]}"; do
  GPU_STATUS[$slot]=0
  GPU_PIDS[$slot]=""
  GPU_TASKS[$slot]=""
done

TOTAL_TASKS=${#TASKS[@]}
STARTED_TASKS=0
FINISHED_TASKS=0

echo "[batch2] root=${ROOT_DIR}"
echo "[batch2] lmms_root=${LMMS_ROOT}"
echo "[batch2] output_path=${OUTPUT_PATH}"
echo "[batch2] model_profile=${MODEL_PROFILE}; model=${MODEL_WRAPPER}"
echo "[batch2] gen_profile=$(resolved_gen_profile)"
echo "[batch2] tasks=${TASK_NAMES}"
echo "[batch2] gpus=${GPU_IDS}"
echo "[batch2] total=${TOTAL_TASKS}"

while [[ ${FINISHED_TASKS} -lt ${TOTAL_TASKS} ]]; do
  for slot in "${!GPUS[@]}"; do
    if [[ "${GPU_STATUS[$slot]}" -eq 1 && -n "${GPU_PIDS[$slot]}" ]]; then
      if ! kill -0 "${GPU_PIDS[$slot]}" 2>/dev/null; then
        echo "[batch2] finish ${GPU_TASKS[$slot]} on GPU ${GPUS[$slot]} (PID ${GPU_PIDS[$slot]})"
        GPU_STATUS[$slot]=0
        GPU_PIDS[$slot]=""
        GPU_TASKS[$slot]=""
        FINISHED_TASKS=$((FINISHED_TASKS + 1))
        echo "[batch2] progress ${FINISHED_TASKS}/${TOTAL_TASKS}"
      fi
    fi
  done

  if [[ ${STARTED_TASKS} -lt ${TOTAL_TASKS} ]]; then
    for slot in "${!GPUS[@]}"; do
      if [[ "${GPU_STATUS[$slot]}" -eq 0 && ${STARTED_TASKS} -lt ${TOTAL_TASKS} ]]; then
        launch_task "${slot}" "${TASKS[$STARTED_TASKS]}"
        STARTED_TASKS=$((STARTED_TASKS + 1))
      fi
    done
  fi

  if [[ ${FINISHED_TASKS} -lt ${TOTAL_TASKS} ]]; then
    sleep "${SLEEP_SECONDS}"
  fi
done

wait
echo "[batch2] all ${TOTAL_TASKS} tasks completed."
