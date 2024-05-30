# cifar100
# text_path='/mnt/ssd_1/gxli/workshop/Longtail/data/cifar100/synthetic/prompts/cifar100.txt'
# save_path='/mnt/ssd_1/gxli/workshop/Longtail/data/cifar100/synthetic/images/'

# cifar10 
# text_path='/mnt/ssd_1/gxli/workshop/Longtail/data/cifar10/synthetic/prompts/cifar10.txt'
# save_path='/mnt/ssd_1/gxli/workshop/Longtail/data/cifar10/synthetic/images/'

text_path=$1
save_path=$2

for i in $(seq 0 1 1)
do
  nohup bash -c "CUDA_VISIBLE_DEVICES=${i} python text2image.py ${i} ${text_path} ${save_path}" > nohup_${i}.out 2>&1 &
done