deepspeed --include localhost:2,3 finetune.py --exp_name "lisa_finetune_test" --test_mode
# deepspeed --include localhost:0,1 finetune.py --exp_name "lisa_finetune_test" --test_mode