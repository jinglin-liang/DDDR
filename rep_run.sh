GPU_LIST=('0' '1' '2' '3')
PID_LIST=()
for i in ${GPU_LIST[*]};
do
    PID_LIST+=('nuLL')
done
echo GPU=${GPU_LIST[*]}


pid_exist(){
    input_id=$1
    key_word='rep_run.sh'
    proc_num=`ps -aux | grep -w $input_id | grep $key_word | grep -v grep | wc -l`
    return $proc_num
}


gpu_monitor(){
    for ((idx=0; idx<${#GPU_LIST[*]}; idx++));
    do
        temp_gpu=${GPU_LIST[${idx}]}
        temp_pid=${PID_LIST[${idx}]}
        pid_exist $temp_pid
        p_num=$?
        if [ $p_num -eq 0 ]
        then
            return $temp_gpu
        fi
    done
    sleep 120s
    return 255
}


date
echo ------------------- start training ------------------------

for METHOD in 'ours' 'target' 'ewc' 'finetune';
do
    if [[ ${METHOD} == 'target' ]]
    then
        WKD=25
    else
        WKD=10
    fi
    for SEED in '2023' '2024' '2025';
    do
        for DATASET in 'cifar100' 'tiny_imagenet';
        do
            for TASK in '5' '10';
            do
                for BETA in '0' '0.5';
                do
                    while true
                    do
                        gpu_monitor
                        free_gpu=$?
                        if [ $free_gpu -ne 255 ]
                        then
                            echo cuda_${free_gpu}_${METHOD}_dataset_${DATASET}_beta_${BETA}_task_${TASK}_seed_${SEED}
                            {
                                CUDA_VISIBLE_DEVICES=${free_gpu} nohup python main.py --dataset ${DATASET} --method ${METHOD} --tasks ${TASK} --beta ${BETA} --seed ${SEED} --w_kd ${WKD} --exp_name reproduction >> outputs/logs/${METHOD}_${DATASET}_beta_${BETA}_task_${TASK}_seed_${SEED}.log 2>&1
                            } &
                            training_pid=$!
                            for ((idx=0; idx<${#GPU_LIST[*]}; idx++));
                            do
                                if [ ${GPU_LIST[$idx]} == $free_gpu ]
                                then
                                    PID_LIST[$idx]=$training_pid
                                    break
                                fi
                            done
                            break
                        fi
                    done
                done
            done
        done
    done
done
wait

echo --------------------- finish -------------------
date
