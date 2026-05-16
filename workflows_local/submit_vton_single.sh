#!/bin/bash
#
# submit_vton_single.sh
# 只跑指定人物和指定衣服范围的 debug 任务
# 每个 job 自动用 socket 找空闲端口，彻底避免端口冲突
#
# 使用方法:
#   PERSON_IDX=0 CLOTH_START=0 CLOTH_END=31 bash submit_vton_single.sh  # 跑人物0的全部衣服
#   PERSON_IDX=5 CLOTH_START=0 CLOTH_END=15 bash submit_vton_single.sh  # 跑人物5的前16件衣服
#   DRY_RUN=1 PERSON_IDX=0 CLOTH_START=0 CLOTH_END=15 bash submit_vton_single.sh  # 只生成脚本不提交
# ============================================================

set -euo pipefail

# 必须设置的人物索引
PERSON_IDX="${PERSON_IDX:-}"
if [ -z "$PERSON_IDX" ]; then
    echo "ERROR: PERSON_IDX must be set"
    echo "Example: PERSON_IDX=0 CLOTH_START=0 CLOTH_END=31 bash submit_vton_single.sh"
    exit 1
fi

# 衣服范围（必须设置）
CLOTH_START="${CLOTH_START:-}"
CLOTH_END="${CLOTH_END:-}"
if [ -z "$CLOTH_START" ] || [ -z "$CLOTH_END" ]; then
    echo "ERROR: CLOTH_START and CLOTH_END must be set"
    echo "Example: PERSON_IDX=0 CLOTH_START=0 CLOTH_END=31 bash submit_vton_single.sh"
    exit 1
fi

DRY_RUN="${DRY_RUN:-0}"

# 可选：任务名称后缀
JOB_SUFFIX="${JOB_SUFFIX:-single}"

JOB_DIR="/tmp/vton_single_jobs_$$"
mkdir -p "$JOB_DIR"
mkdir -p logs

echo "============================================"
echo "  VTON Single Person Submission"
echo "  Person: ${PERSON_IDX}"
echo "  Clothes range: ${CLOTH_START} - ${CLOTH_END}"
echo "  Total clothes: $((CLOTH_END - CLOTH_START + 1))"
echo "  Port: auto (OS-assigned free port)"
if [ "$DRY_RUN" = "1" ]; then
    echo "  MODE: DRY RUN (script in ${JOB_DIR})"
else
    echo "  MODE: LIVE (will sbatch)"
fi
echo "============================================"
echo ""

SCRIPT="${JOB_DIR}/vton_p${PERSON_IDX}_${JOB_SUFFIX}.sh"

cat > "$SCRIPT" << 'SCRIPTEND'
#!/bin/bash
#SBATCH -p debug
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
SCRIPTEND

cat >> "$SCRIPT" << SCRIPTEND
#SBATCH -J vton_p${PERSON_IDX}_${JOB_SUFFIX}
#SBATCH -o logs/job_vton_p${PERSON_IDX}_${JOB_SUFFIX}_%j.out
#SBATCH -e logs/job_vton_p${PERSON_IDX}_${JOB_SUFFIX}_%j.err

export PERSON_IDX=${PERSON_IDX}
export CLOTH_START=${CLOTH_START}
export CLOTH_END=${CLOTH_END}
SCRIPTEND

cat >> "$SCRIPT" << 'SCRIPTEND'

cd /hpc2hdd/home/dsaa2012_017/comfyui/ComfyUI || exit 1
source /hpc2hdd/home/dsaa2012_017/miniconda3/bin/activate py_312
module load cuda/12.8

COMFY_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); p=s.getsockname()[1]; s.close(); print(p)")
export COMFY_PORT
echo "Assigned free port: ${COMFY_PORT}"

echo "$(date): Starting ComfyUI on port ${COMFY_PORT}..."
python main.py --listen 0.0.0.0 --port ${COMFY_PORT} \
    --preview-method auto --enable-manager > comfyui_${COMFY_PORT}.log 2>&1 &
COMFY_PID=$!
echo "ComfyUI PID = $COMFY_PID"

echo "Waiting for ComfyUI to start..."
sleep 60

READY=false
for i in {1..60}; do
    if curl -sf "http://127.0.0.1:${COMFY_PORT}/object_info" > /dev/null 2>&1; then
        READY=true
        echo "ComfyUI ready on port ${COMFY_PORT}"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 10
done

if [ "$READY" = "false" ]; then
    echo "ERROR: ComfyUI failed to start on port ${COMFY_PORT}"
    kill $COMFY_PID 2>/dev/null || true
    exit 1
fi

cd /hpc2hdd/home/dsaa2012_017/comfyui || exit 1
echo "$(date): Running VTON batch: person=${PERSON_IDX}, cloth=${CLOTH_START}-${CLOTH_END}"
python run_vton_batch.py

echo "$(date): Monitoring ComfyUI queue..."
TASK_COMPLETE=false
while ! $TASK_COMPLETE; do
    QUEUE_STATUS=$(curl -s http://127.0.0.1:${COMFY_PORT}/queue)
    RUNNING=$(echo $QUEUE_STATUS | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('queue_running', [])))" 2>/dev/null || echo "0")
    PENDING=$(echo $QUEUE_STATUS | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('queue_pending', [])))" 2>/dev/null || echo "0")
    TOTAL=$((RUNNING + PENDING))
    echo "[$(date '+%H:%M:%S')] Running: $RUNNING, Pending: $PENDING, Total: $TOTAL"

    if [ "$TOTAL" = "0" ] && [ "$RUNNING" = "0" ]; then
        sleep 10
        QUEUE_STATUS=$(curl -s http://127.0.0.1:${COMFY_PORT}/queue)
        RUNNING=$(echo $QUEUE_STATUS | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('queue_running', [])))" 2>/dev/null || echo "0")
        PENDING=$(echo $QUEUE_STATUS | python3 -c "import sys,json; data=json.load(sys.stdin); print(len(data.get('queue_pending', [])))" 2>/dev/null || echo "0")
        if [ "$RUNNING" = "0" ] && [ "$PENDING" = "0" ]; then
            echo "All tasks completed!"
            TASK_COMPLETE=true
            break
        fi
    fi

    if ! kill -0 $COMFY_PID 2>/dev/null; then
        echo "ComfyUI process exited unexpectedly"
        break
    fi
    sleep 15
done

curl -s http://127.0.0.1:${COMFY_PORT}/history > vton_history_p${PERSON_IDX}_${JOB_SUFFIX}.json 2>/dev/null || true
kill $COMFY_PID 2>/dev/null || true
wait $COMFY_PID 2>/dev/null || true
echo "$(date): Job finished"
SCRIPTEND

chmod +x "$SCRIPT"

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY] Created: ${SCRIPT}"
    echo ""
    echo "To submit manually, run:"
    echo "  sbatch ${SCRIPT}"
else
    echo "Submitting job..."
    sbatch "$SCRIPT"
    echo ""
    echo "Job submitted! Check status with:"
    echo "  squeue -u $USER"
    echo "  tail -f logs/job_vton_p${PERSON_IDX}_${JOB_SUFFIX}_*.out"
fi

echo ""
echo "============================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY RUN done. Script in: ${SCRIPT}"
else
    echo "  Job submitted for person ${PERSON_IDX}, clothes ${CLOTH_START}-${CLOTH_END}"
fi
echo "============================================"
