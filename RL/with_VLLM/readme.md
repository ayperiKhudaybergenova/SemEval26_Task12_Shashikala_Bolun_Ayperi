# GRPO Training with Unsloth & vLLM on bwHPC

This project implements Abductive Event Reasoning training using **Unsloth** (for efficient LoRA/QLoRA) and **vLLM** (for fast generation) via the GRPO algorithm. The setup is optimized for multi-GPU nodes (e.g., H100/A100) on bwHPC clusters (like bwUniCluster 3.0).

## 1. Container Preparation (Enroot)

Before running the scripts, you must set up the container environment. bwHPC clusters use **Enroot** (via Pyxis) for container management.

1. **Login** to the cluster.
2. **Import** the stable Unsloth image from Docker Hub.
3. **Create** the container and name it `unsloth1` (this name must match the `#SBATCH --container-name` directive in `runVLLM.sh`).

```bash
# 1. Pull the image
enroot import docker://unsloth/unsloth:stable

# 2. Create the container named 'unsloth1'
# Note: The imported file will be named 'unsloth+unsloth+stable.sqsh'
enroot create --name unsloth1 unsloth+unsloth+stable.sqsh

# 3. Verify creation
enroot list

```

---

## 2. Configuration

### A. Path Setup (`runVLLM.sh`)

You must define where your project files reside.

* Open `runVLLM.sh`.
* Locate : `export PROJECT_DIR="$HOME/AER"`.
* **Action:** Change `"$HOME/AER"` to the actual path of your project directory.
* 
*Tip:* For large datasets or checkpoints, consider using a **Workspace** instead of `$HOME` to avoid quota limits.





### B. Model & Data (`trainRLVLLM.py`)

* **Model Selection:** Modify `MODEL_ID` to change the base model (e.g., `Qwen/Qwen3-32B`).
* **4-bit Quantization:**
* For large models (like 32B or 70B), ensure `load_in_4bit=True` is set in the `FastLanguageModel.from_pretrained` call. This significantly reduces VRAM usage.



---

## 3. Handling OOM (Out of Memory)

If you encounter CUDA Out of Memory (OOM) errors, you need to balance the `PER_DEVICE_BATCH_SIZE` and `GRAD_ACCUM_STEPS` in `trainRLVLLM.py`.

**Goal:** Keep the **Total Batch Size** constant to maintain training stability, while reducing the memory load per GPU.

**Adjustment Strategy:**

1. **Decrease** `PER_DEVICE_BATCH_SIZE`.
2. **Increase** `GRAD_ACCUM_STEPS` proportionally.

**Example Scenario (2 GPUs):**

| Scenario | Per Device Batch | Grad Accum Steps | GPUs | **Total Batch Size** | Memory Usage |
| --- | --- | --- | --- | --- | --- |
| **Fix 1** | 4 | 2 | 2 | **16** | High |
| **Current** | 2 | 4 | 2 | **16** | Medium |
| **Fix 2** | 1 | 8 | 2 | **16** | Low |

*Note: `PER_DEVICE_BATCH_SIZE` for vLLM usually needs to be smaller than standard training because vLLM reserves memory for the KV cache.*

*Note: as long as the gpu(s) can go full power, it's better to have lower `PER_DEVICE_BATCH_SIZE`, to avoid OOM or deadlock caused by VRAM shortage*

*Tip*: 

srun --jobid=<your_Job_ID> --pty /bin/bash

watch -n 1 nvidia-smi

---

## 4. Execution & Testing

### Step 1: Dry Run on Dev Partition

Always test on a development partition before submitting a long job to ensure code validity and container mounting.

1. Open `runVLLM.sh`.
2. Modify the `#SBATCH` directives for the test:
* 
**Partition:** Change to a dev queue (e.g., `dev_gpu_h100` or `dev_gpu_a100_il` depending on your cluster).


* 
**Time:** Set a short limit, e.g., `#SBATCH --time=00:30:00` (Dev queues often have a 30-min limit).




3. **Submit:**
```bash
sbatch runVLLM.sh

```


4. **Check Logs:** Monitor the `logs/` directory. Ensure the container loads, `unsloth` imports successfully, and training starts without immediate crash.

### Step 2: Full Training

Once the test passes:

1. Revert `runVLLM.sh` settings:
* **Partition:** Set to the production queue (e.g., `gpu_h100` or `gpu_a100`).
* **Time:** Set to desired duration (e.g., `36:00:00`).


2. **Submit:**
```bash
sbatch runVLLM.sh

```



---

## 5. Script Overview

| File | Description |
| --- | --- |
| `runVLLM.sh` | **SLURM Entry Point.** Requests GPUs, mounts the container (mapping host paths to container paths), sets environment variables (NCCL, Python isolation), and launches the training script via `accelerate`. |
| `trainRLVLLM.py` | **Training Logic.** <br> <br>1. Loads model with Unsloth (4-bit).<br> <br>2. Formats data with "Structure-Aware" truncation.<br> <br>3. Defines Reward Functions.<br> <br>4. Runs GRPOTrainer with vLLM generation enabled. |

## 6. Troubleshooting Notes

* **Log Spam:** The script sets `export ENROOT_ENTRYPOINT=0` to disable the default NVIDIA entrypoint, which reduces log noise.
* **Network/NCCL:** `NCCL_P2P_DISABLE=1` and `NCCL_IB_DISABLE=1` are set to prevent deadlocks common in containerized SLURM environments without specific interconnect configurations.
* **Cache:** `VLLM_DISABLE_COMPILE_CACHE=1` is set to prevent locking issues on shared filesystems.
