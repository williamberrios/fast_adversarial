# +
# 4eps evaluation
#python main_fast.py ~/imagenet --config configs_fast_4px_evaluate.yml --output_prefix eval_4px --resume trained_models/fast_adv_phase3_eps4_step5_eps4_repeat1/model_best.pth.tar --evaluate --restarts 10
# -

# 2eps evaluation 
python main_fast_adv_MVIT.py ../../Dataset/ILSVRC/Data/imagenet --config configs/configs_fast_2px_evaluate.yml --output_prefix eval_2px --resume ./trained_models/fast_adv_phase1_2px_step2_eps2_repeat1/model_best.pth.tar --evaluate --restarts 10 |tee out_files/eval_2px.out
