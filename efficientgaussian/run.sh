# python ./importance_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/importance/ --save_ply >> ./proj/importance.txt

# python ./mag_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/magnitude/ --save_ply >> ./proj/mag.txt

# python ./struc_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/structured/ --save_ply >> ./proj/struc.txt

# python ./train_eval.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free// -m ./log_dir/strike_freedom_p/origin/ --save_ply >> ./proj/origin.txt

# python render_360.py -m ./log_dir/new_g/origin/ -s ./custom_data/new_g/ --config ./configs/efficient-3dgs.yaml
# python render_360.py -m ./log_dir/new_g/magnitude/ -s ./custom_data/new_g/ --config ./configs/efficient-3dgs.yaml
# python render_360.py -m ./log_dir/new_g/importance/ -s ./custom_data/new_g/ --config ./configs/efficient-3dgs.yaml

#python ./mag_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/fine_tuning/magnitude0.4/ --save_ply >> ./proj/f_mag0.4.txt
#python ./mag_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/new_g/ -m ./log_dir/new_g/n_magnitude0.4/ --save_ply >> ./proj/n_mag0.4re.txt


# python ./d_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/d_train --save_ply >> ./proj/sd_train.txt
# python ./d_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/new_g/ -m ./log_dir/new_g/d_train --save_ply >> ./proj/nd_train.txt
# magnitude 0.3 0.4돌려보기 importance 파인튜닝 2번 시도

# python ./importance_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/new_g/ -m ./log_dir/new_g/nimportance/ --save_ply >> ./proj/nn_importance.txt
# python ./importance_prune_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/nimportance/ --save_ply >> ./proj/nimportance.txt

python render_360.py -m ./log_dir/new_g/strike_freedom_p/d2_train/ -s ./custom_data/s_free/ --config ./configs/efficient-3dgs.yaml
python render_360.py -m ./log_dir/new_g/d2_train/ -s ./custom_data/new_g/ --config ./configs/efficient-3dgs.yaml

# python ./d_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/s_free/ -m ./log_dir/strike_freedom_p/d2_train --save_ply >> ./proj/sd2_train.txt
# python ./d_train.py --config ./configs/efficient-3dgs.yaml -s ./custom_data/new_g/ -m ./log_dir/new_g/d2_train --save_ply >> ./proj/nd2_train.txt