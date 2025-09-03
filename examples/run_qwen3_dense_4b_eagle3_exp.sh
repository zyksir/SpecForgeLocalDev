export SHARED_CACHE_PREFIX=/root/.cache/user_artifacts/
export MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
export DATASET_PATH=$SHARED_CACHE_PREFIX/Qwen3-4B-Instruct-2507/dataset/
export CACHE_DIR=$SHARED_CACHE_PREFIX/Qwen3-4B-Instruct-2507/cache
export HIDDEN_STATES_DIR=$SHARED_CACHE_PREFIX/Qwen3-4B-Instruct-2507/hidden_states
export OUTPUT_DIR=$SHARED_CACHE_PREFIX/Qwen3-4B-Instruct-2507/outputs/
export CHAT_TEMPLATE=qwen
export MAX_LENGTH=2048
export DRAFT_MODEL_CONFIG=./configs/qwen3-4B-eagle3.json

hf download $MODEL_PATH

python scripts/prepare_data.py --dataset sharegpt --output_path $DATASET_PATH
python scripts/prepare_data.py --dataset ultrachat --output_path $DATASET_PATH

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
export NUM_GPUS=4
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --nproc_per_node $NUM_GPUS \
    scripts/train_eagle3_online_v2.py \
    --target-model-path $MODEL_PATH \
    --model-path $MODEL_PATH \
    --embedding-key model.embed_tokens.weight \
    --lm-head-key model.embed_tokens.weight \
    --draft-model-config $DRAFT_MODEL_CONFIG \
    --train-data-path $DATASET_PATH/sharegpt_ultrachat_train.jsonl \
    --eval-data-path $DATASET_PATH/sharegpt_ultrachat_test.jsonl \
    --tp-size $NUM_GPUS \
    --output-dir $OUTPUT_DIR \
    --num-epochs 10 \
    --batch-size 1 \
    --learning-rate 5e-5 \
    --max-length $MAX_LENGTH \
    --chat-template $CHAT_TEMPLATE \
    --cache-dir $CACHE_DIR \
    --mem-frac=0.4 --wandb-project eagle4  --wandb-name base-dense-4b-target-dense --wandb

config_list=(
    "1,3,1,4"
    "1,7,10,60"
)
CUDA_VISIBLE_DEVICES=4,5,6,7 python3 benchmarks/bench_model_speedup.py \
    --model-path Qwen/Qwen3-4B-Instruct-2507 \
    --speculative-draft-model-path /root/.cache/user_artifacts/Qwen3-4B-Instruct-2507/outputs/epoch_4 \
    --port 20002 \
    --trust-remote-code \
    --mem-fraction-static 0.8 \
    --attention-backend fa3 \
    --config-list "${config_list[@]}" \
    --benchmark-list mtbench:80 \
    --output Qwen3-4B-Instruct-2507_Eagle3-300k_result.jsonl