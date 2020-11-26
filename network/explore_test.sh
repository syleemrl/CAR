lb=$1
n=0
print_file="/home/seyoung/Projects/CAR/network/output/test_result${lb}"
base_file="/home/seyoung/Projects/CAR/network/output/punch_ue_2d_test"
target_file="/home/seyoung/Projects/CAR/network/output/punch_ue_2d_${lb}"
for iter in $(seq 1 50);
do 
  target_file_full="${target_file}_${n}"
  target_file_part="punch_ue_2d_${lb}_${n}"
  cp -r ${base_file} ${target_file_full}
  cd ../utils
  python3 tensorflow_rename_variables.py --checkpoint_dir=${target_file_full} --replace_from=punch_ue_2d_test --replace_to=${target_file_part}
  cd ../network
  python3 ppo.py --parametric --adaptive --ref=punch_ue.bvh --test_name=${target_file_part} --pretrain="${target_file_full}/network-0" --nslaves=8 --lb=${lb} --test_path=${print_file}
  rm -r ${target_file_full}
  n=$((n+1))
done
