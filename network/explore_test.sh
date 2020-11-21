visit=1
explore=$1
egreedy=$2
hard=$3
n=0
print_file="/home/sonic/Projects/CAR/network/output/test_result${visit}${explore}${egreedy}${hard}"
base_file="/home/sonic/Projects/CAR/network/output/punch_ue_t2"
target_file="/home/sonic/Projects/CAR/network/output/punch_ue_t${visit}${explore}${egreedy}${hard}"
for iter in $(seq 1 50);
do 
  target_file_full="${target_file}_${n}"
  target_file_part="punch_ue_t${visit}${explore}${egreedy}${hard}_${n}"
  cp -r ${base_file} ${target_file_full}
  cd ../utils
  python3 tensorflow_rename_variables.py --checkpoint_dir=${target_file_full} --replace_from=punch_ue_t2 --replace_to=${target_file_part}
  cd ../network
  python3 ppo.py --parametric --adaptive --ref=punch_ue.bvh --test_name=${target_file_part} --pretrain="${target_file_full}/network-0" --nslaves=8 --explore=${explore} --visit=${visit} --egreedy=${egreedy} --hard=${hard} --exploration_test_print=${print_file}
  rm -r ${target_file_full}
  n=$((n+1))
done
