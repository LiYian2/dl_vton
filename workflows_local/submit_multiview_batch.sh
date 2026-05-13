#!/bin/bash
#
# submit_multiview_batch.sh
# 48 张图，每 job 处理 3 张，共 16 个 debug 任务
# 每个 job 自动找空闲端口，间隔 10min 提交
#
# 使用方法:
#   bash submit_multiview_batch.sh
#   DRY_RUN=1 bash submit_multiview_batch.sh
# ============================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"

IMAGES_PER_JOB=3
TOTAL_IMAGES=48
TOTAL_JOBS=$((TOTAL_IMAGES / IMAGES_PER_JOB))

JOB_DIR="/tmp/multiview_batch_jobs_$$"
mkdir -p "$JOB_DIR"
mkdir -p logs

echo "============================================"
echo "  Multiview Batch Submission"
echo "  Images per job: ${IMAGES_PER_JOB}"
echo "  Total images: ${TOTAL_IMAGES}"
echo "  Total jobs: ${TOTAL_JOBS}"
echo "  Port: auto (OS-assigned free port)"
if [ "$DRY_RUN" = "1" ]; then
    echo "  MODE: DRY RUN (scripts in ${JOB_DIR})"
else
    echo "  MODE: LIVE (will sbatch)"
fi
echo "============================================"
echo ""

if [ "$DRY_RUN" != "1" ]; then
    echo "Submitting ${TOTAL_JOBS} jobs with 10 min interval..."
    echo ""
fi

SUBMITTED=0

for JOB_IDX in $(seq 0 $((TOTAL_JOBS - 1))); do
    START_IDX=$((JOB_IDX * IMAGES_PER_JOB))
    END_IDX=$((START_IDX + IMAGES_PER_JOB - 1))

    SCRIPT="${JOB_DIR}/multiview_job_${JOB_IDX}.sh"

    cat > "$SCRIPT" << 'SCRIPTEND'
#!/bin/bash
#SBATCH -p debug
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
SCRIPTEND

    cat >> "$SCRIPT" << SCRIPTEND
#SBATCH -J mv_${START_IDX}_${END_IDX}
#SBATCH -o logs/job_multiview_${START_IDX}_${END_IDX}_%j.out
#SBATCH -e logs/job_multiview_${START_IDX}_${END_IDX}_%j.err

export START_IDX=${START_IDX}
export END_IDX=${END_IDX}
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
echo "$(date): Running Multiview batch: idx=${START_IDX}-${END_IDX}"
python run_multiview_batch.py

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

curl -s http://127.0.0.1:${COMFY_PORT}/history > multiview_history_${START_IDX}_${END_IDX}.json 2>/dev/null || true
kill $COMFY_PID 2>/dev/null || true
wait $COMFY_PID 2>/dev/null || true
echo "$(date): Job finished"
SCRIPTEND

    chmod +x "$SCRIPT"

    if [ "$DRY_RUN" = "1" ]; then
        echo "[DRY] Created: ${SCRIPT}"
    else
        sbatch "$SCRIPT" && SUBMITTED=$((SUBMITTED + 1))
        echo "[$((JOB_IDX + 1))/${TOTAL_JOBS}] Submitted: idx=${START_IDX}-${END_IDX}"
        if [ $JOB_IDX -lt $((TOTAL_JOBS - 1)) ]; then
            echo "  Waiting 10 min before next submission..."
            sleep 600
        fi
    fi
done

echo ""
echo "============================================"
if [ "$DRY_RUN" = "1" ]; then
    echo "  DRY RUN done. Scripts in: ${JOB_DIR}"
else
    echo "  Submitted ${SUBMITTED}/${TOTAL_JOBS} jobs"
    echo "  Scripts in: ${JOB_DIR}"
fi
echo "============================================"
