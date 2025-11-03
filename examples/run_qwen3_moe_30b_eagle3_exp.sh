

# export PERSIST_DIR=/root/.cache/user_artifacts/
export PERSIST_DIR=/data/yikai
export MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507
export MODEL_NAME=Qwen3-30B-A3B-Instruct-2507

export RAW_DATASET_PATH=$PERSIST_DIR/dataset/
export GENERATED_DATASET_PATH=$PERSIST_DIR/$MODEL_NAME/generated_dataset
export CACHE_DIR=$PERSIST_DIR/$MODEL_NAME/cache
export OUTPUT_DIR=$PERSIST_DIR/$MODEL_NAME/outputs/
export CHAT_TEMPLATE=qwen
export MAX_LENGTH=2048
export DRAFT_MODEL_CONFIG=./configs/qwen3-30B-A3B-eagle3.json

hf download $MODEL_PATH
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
# hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

python scripts/prepare_data.py --dataset sharegpt --output-path $RAW_DATASET_PATH


for i in {1..8}; do
    CUDA_VISIBLE_DEVICES=$((i-1)) python3 -m sglang.launch_server \
        --model $MODEL_PATH \
        --cuda-graph-bs 1 2 4 8 16 32 64 128 256 \
        --mem-frac=0.8 --port $((30000 + i)) &
done

python scripts/generate_data_by_target.py \
    --model-name $MODEL_PATH \
    --raw-data-file $RAW_DATASET_PATH/sharegpt_train.jsonl \
    --output-dir $GENERATED_DATASET_PATH/sharegpt \
    --max-concurrency 256 \
    --num-per-shard 50000 \
    --max-tokens $MAX_LENGTH \
    --is-reasoning-model \
    --server-address-port 127.0.0.1:30001 127.0.0.1:30002 127.0.0.1:30003 127.0.0.1:30004 127.0.0.1:30005 127.0.0.1:30006 127.0.0.1:30007 127.0.0.1:30008


export DATASET_PATH=$PERSIST_DIR/$MODEL_NAME/generated_dataset/
cat $DATASET_PATH/sharegpt/shard*.jsonl > $DATASET_PATH/sharegpt_train.jsonl
python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/sharegpt_train.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1

export NUM_GPUS=4
export DATASET_PATH=$PERSIST_DIR/$MODEL_NAME/generated_dataset/
export OUTPUT_DIR=$PERSIST_DIR/Qwen3-30B-A3B-Instruct-2507/moe_wei_dense-fake-residual/
export DRAFT_MODEL_CONFIG=./configs/qwen3-30B-A3B-eagle3-wei-dense.json
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_sgl_online.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/sharegpt_train.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 4e-4 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --resume \
    --mem-frac=0.4 \
    --total-steps=800000 \
    --save-total-limit=8 \
    --warmup-ratio=0.015 \
    --dist-timeout=10 \
    --wandb-project eagle4 \
    --wandb-name base-moe-30b-wei-dense-fake-residual \
    --report-to wandb

config_list=(
    "4,7,10,60"
    "4,3,1,4"
)
for i in {0..4}; do
    CUDA_VISIBLE_DEVICES=$i python3 benchmarks/bench_model_speedup.py \
        --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --speculative-draft-model-path $PERSIST_DIR/Qwen3-30B-A3B-Instruct-2507/moe_wei_dense-fake-residual/step_$(((i+4)*7529)) \
        --port $((20000 + i)) \
        --trust-remote-code \
        --mem-fraction-static 0.8 \
        --attention-backend fa3 \
        --config-list "${config_list[@]}" \
        --benchmark-list mtbench:80 gsm8k:200 humaneval:200 math500:200 \
        --output Qwen3-30B-A3B-Wei-DENSE-ADAPTIVE_Eagle3-sharegpt_result_step_$(((i+2)*7529)).jsonl &
done

