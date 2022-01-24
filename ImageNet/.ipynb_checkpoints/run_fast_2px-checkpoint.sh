DATA=../../Dataset/ILSVRC/Data/imagenet

NAME=2px

CONFIG1=configs/configs_fast_${NAME}_phase1.yml
CONFIG2=configs/configs_fast_${NAME}_phase2.yml
CONFIG3=configs/configs_fast_${NAME}_phase3.yml
CONFIG4=configs/configs_fast_${NAME}_phase4.yml
CONFIG5=configs/configs_fast_${NAME}_phase5.yml
CONFIG6=configs/configs_fast_${NAME}_phase6.yml

PREFIX1=fast_adv_phase1_${NAME}
PREFIX2=fast_adv_phase2_${NAME}
PREFIX3=fast_adv_phase3_${NAME}
PREFIX4=fast_adv_phase4_${NAME}
PREFIX5=fast_adv_phase5_${NAME}
PREFIX6=fast_adv_phase6_${NAME}

END1=./trained_models/fast_adv_phase1_${NAME}_step2_eps2_repeat1/checkpoint_epoch4.pth.tar


OUT1=out_files/fast_train_phase1_${NAME}.out
OUT2=out_files/fast_train_phase2_${NAME}.out
OUT3=out_files/fast_train_phase3_${NAME}.out
OUT4=out_files/fast_train_phase3_${NAME}.out
OUT5=out_files/fast_train_phase3_${NAME}.out
OUT6=out_files/fast_train_phase3_${NAME}.out

#python -u main_fast_adv_MVIT.py $DATA -c $CONFIG1 --output_prefix $PREFIX1  | tee $OUT1
python -u main_fast_adv_MVIT.py $DATA -c $CONFIG2 --output_prefix $PREFIX2 --resume $END1 | tee $OUT2
#python -u main_fast_adv_MVIT.py $DATA -c $CONFIG3 --output_prefix $PREFIX3 --resume $END2 | tee $OUT3
#python -u main_fast_adv_MVIT.py $DATA -c $CONFIG4 --output_prefix $PREFIX4  --resume $END3| tee $OUT4
#python -u main_fast_adv_MVIT.py $DATA -c $CONFIG5 --output_prefix $PREFIX5  --resume $END4| tee $OUT5
#python -u main_fast_adv_MVIT.py $DATA -c $CONFIG6 --output_prefix $PREFIX6  --resume $END5| tee $OUT6
