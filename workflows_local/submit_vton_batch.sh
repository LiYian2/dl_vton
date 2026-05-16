#!/bin/bash
#
# submit_vton_batch.sh
# 将 24 个人物 x 2 批衣服（每批16件）拆成 48 个 debug 任务提交
# 每个 job 自动用 socket 找空闲端口，彻底避免端口冲突
#
# 使用方法:
#   bash submit_vton_batch.sh           # 提交所有任务
#   DRY_RUN=1 bash submit_vton_batch.sh # 只生成脚本不提交
# ============================================================

set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
#原本计划每批 16 件，但为了更快完成测试，这里改成 32 件，减少总 job 数量
CLOTH_PER_JOB=16
NUM_PERSONS=24
NUM_HALVES=2
TOTAL_JOBS=$((NUM_PERSONS * NUM_HALVES))

JOB_DIR="/tmp/vton_batch_jobs_$$"
mkdir -p "$JOB_DIR"
mkdir -p logs

echo "============================================"
echo "  VTON Batch Submission"
echo "  Persons: 0-23 (${NUM_PERSONS} total)"
echo "  Clothes per job: ${CLOTH_PER_JOB}"
echo "  Halves per person: ${NUM_HALVES}"
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

JOB_NUM=0
SUBMITTED=0

for PERSON_IDX in $(seq 0 $((NUM_PERSONS - 1))); do
    for HALF in $(seq 0 $((NUM_HALVES - 1))); do
        CLOTH_START=$((HALF * CLOTH_PER_JOB))
        CLOTH_END=$((CLOTH_START + CLOTH_PER_JOB - 1))

        SCRIPT="${JOB_DIR}/vton_p${PERSON_IDX}_h${HALF}.sh"

        cat > "$SCRIPT" << 'SCRIPTEND'
#!/bin/bash
#SBATCH -p debug
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
SCRIPTEND

        cat >> "$SCRIPT" << SCRIPTEND
#SBATCH -J vton_p${PERSON_IDX}_h${HALF}
#SBATCH -o logs/job_vton_p${PERSON_IDX}_h${HALF}_%j.out
#SBATCH -e logs/job_vton_p${PERSON_IDX}_h${HALF}_%j.err

export PERSON_IDX=${PERSON_IDX}
export CLOTH_START=${CLOTH_START}
export CLOTH_END=${CLOTH_END}
export HALF=${HALF}
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

curl -s http://127.0.0.1:${COMFY_PORT}/history > vton_history_p${PERSON_IDX}_h${HALF}.json 2>/dev/null || true
kill $COMFY_PID 2>/dev/null || true
wait $COMFY_PID 2>/dev/null || true
echo "$(date): Job finished"
SCRIPTEND

        chmod +x "$SCRIPT"

        if [ "$DRY_RUN" = "1" ]; then
            echo "[DRY] Created: ${SCRIPT}"
        else
            sbatch "$SCRIPT" && SUBMITTED=$((SUBMITTED + 1))
            echo "[$((JOB_NUM + 1))/${TOTAL_JOBS}] Submitted: p${PERSON_IDX}_h${HALF}  cloth=${CLOTH_START}-${CLOTH_END}"
            if [ $JOB_NUM -lt $((TOTAL_JOBS - 1)) ]; then
                echo "  Waiting 10 min before next submission..."
                sleep 600
            fi
        fi

        JOB_NUM=$((JOB_NUM + 1))
    done
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
