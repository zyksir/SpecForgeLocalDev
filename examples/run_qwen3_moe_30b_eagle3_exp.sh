# export SHARED_CACHE_PREFIX=/root/.cache/user_artifacts/
# export SHARED_CACHE_PREFIX=/data/yikai
export SHARED_CACHE_PREFIX=~/.cache/huggingface/
export MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507
export DATASET_PATH=$SHARED_CACHE_PREFIX/dataset/
export CACHE_DIR=$SHARED_CACHE_PREFIX/Qwen3-30B-A3B-Instruct-2507/cache
export HIDDEN_STATES_DIR=$SHARED_CACHE_PREFIX/Qwen3-30B-A3B-Instruct-2507/hidden_states
export OUTPUT_DIR=$SHARED_CACHE_PREFIX/Qwen3-30B-A3B-Instruct-2507/outputs/
export CHAT_TEMPLATE=qwen
export MAX_LENGTH=2048
export DRAFT_MODEL_CONFIG=./configs/qwen3-30B-A3B-eagle3.json

hf download $MODEL_PATH
hf download Aeala/ShareGPT_Vicuna_unfiltered --repo-type dataset
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

python scripts/prepare_data.py --dataset sharegpt --output-path $DATASET_PATH --split-eval
python scripts/prepare_data.py --dataset ultrachat --output-path $DATASET_PATH --split-eval

cat $DATASET_PATH/sharegpt_train.jsonl $DATASET_PATH/ultrachat_train.jsonl > $DATASET_PATH/sharegpt_ultrachat_train.jsonl
cat $DATASET_PATH/sharegpt_test.jsonl $DATASET_PATH/ultrachat_test.jsonl > $DATASET_PATH/sharegpt_ultrachat_test.jsonl

python scripts/build_eagle3_dataset_cache.py \
    --target-model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat_train.jsonl \
    --eval-data-path $DATASET_PATH/sharegpt_ultrachat_test.jsonl \
    --cache-dir $CACHE_DIR \
    --chat-template $CHAT_TEMPLATE \
    --max-length $MAX_LENGTH \
    --view-train-data 1

export NUM_GPUS=2 # 8
export OUTPUT_DIR=$SHARED_CACHE_PREFIX/Qwen3-30B-A3B-Instruct-2507/moe_draft_outputs_aux/
export DRAFT_MODEL_CONFIG=./configs/qwen3-30B-A3B-MoEHead-eagle3.json
CUDA_VISIBLE_DEVICES=1,2 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_sgl_online.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat_train.jsonl \
    --eval-data-path $DATASET_PATH/sharegpt_ultrachat_test.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --draft-attention-backend flex_attention \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --resume \
    --mem-frac=0.4 \
    --total-steps=800000 \
    --warmup-ratio=0.015 \
    --dist-timeout=10 \
    --wandb-project eagle4 \
    --wandb-name base-moe-30b-draft-moe-aux-loss \
    --report-to wandb

config_list=(
    "1,3,1,4"
    "1,7,10,60"
)
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 benchmarks/bench_model_speedup.py \
    --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --speculative-draft-model-path /root/.cache/user_artifacts/Qwen3-30B-A3B-Instruct-2507/outputs/epoch_4 \
    --port 20001 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --attention-backend fa3 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 \
    --output Qwen3-30B-A3B-Instruct-2507_Eagle3-300k_result.jsonl