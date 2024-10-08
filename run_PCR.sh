export OMP_NUM_THREADS=1

ALPHA=(0.0 0.1 0.3 0.5 0.7 0.9 1.0)

for i in ${!ALPHA[@]}; do
    CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
    --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train_pcr_hiqa.py --alpha ${ALPHA[$i]} --output "output/Ev2_PCRHIQA_${ALPHA[$i]}" 
    ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
done