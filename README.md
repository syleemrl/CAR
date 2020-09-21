# Learning a family of motor skills from a single motion clip

### install 

> 1. cuda 10.2 설치
> 2. CAR 폴더에서 ./install.sh
> 3. export PYTHONPATH=$PYTHONPATH:/PATH/TO/CAR/network 를 터미널 설정 파일(bashrc, zshrc 등)에 추가

### render

> 1. cd ./build/render
> 2. ./render --type=sim --ref=bvh_name.bvh --network=network_name/network-0
>> * 이 외의 argument 옵션은 ./render/main.cpp 파일 참고

### train

> 1. cd ./network
> 2. python3 ppo.py --ref=bvh_name.bvh --test_name=test_name --pretrain=output/test_name/network-0
>> * pretrain이 있을 경우 test_name은 같은 이름으로 설정
>> * 이 외의 argument 옵션은 network/ppo.py 파일 참고
