nohup python -u main.py --gpu 0 --imbanlance_rate 0.005 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy drop >log/cifar10_IR200_basic_drop.log 2>&1 &
nohup python -u main.py --gpu 1 --imbanlance_rate 0.01 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy drop >log/cifar10_IR100_basic_drop.log 2>&1 &
nohup python -u main.py --gpu 2 --imbanlance_rate 0.02 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy drop >log/cifar10_IR50_basic_drop.log 2>&1 &
nohup python -u main.py --gpu 3 --imbanlance_rate 0.1 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy drop >log/cifar10_IR10_basic_drop.log 2>&1 &

nohup python -u main.py --gpu 0 --imbanlance_rate 0.005 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy pos_neg >log/cifar10_IR200_basic_pos_neg.log 2>&1 &
nohup python -u main.py --gpu 1 --imbanlance_rate 0.01 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy pos_neg >log/cifar10_IR100_basic_pos_neg.log 2>&1 &
nohup python -u main.py --gpu 2 --imbanlance_rate 0.02 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy pos_neg >log/cifar10_IR50_basic_pos_neg.log 2>&1 &
nohup python -u main.py --gpu 3 --imbanlance_rate 0.1 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy pos_neg >log/cifar10_IR10_basic_pos_neg.log 2>&1 &

nohup python -u main.py --gpu 0 --imbanlance_rate 0.005 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy neg_only >log/cifar10_IR200_basic_neg.log 2>&1 &
nohup python -u main.py --gpu 1 --imbanlance_rate 0.01 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy neg_only >log/cifar10_IR100_basic_neg.log 2>&1 &
nohup python -u main.py --gpu 2 --imbanlance_rate 0.02 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy neg_only >log/cifar10_IR50_basic_neg.log 2>&1 &
nohup python -u main.py --gpu 3 --imbanlance_rate 0.1 -a resnet32 --dataset cifar10 --num_classes 10 --lr 0.01 --loss_strategy neg_only >log/cifar10_IR10_basic_neg.log 2>&1 &



nohup python -u main.py --gpu 0 --imbanlance_rate 0.005 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy drop >log/cifar100_IR200_basic_drop.log 2>&1 &
nohup python -u main.py --gpu 1 --imbanlance_rate 0.01 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy drop >log/cifar100_IR100_basic_drop.log 2>&1 &
nohup python -u main.py --gpu 2 --imbanlance_rate 0.02 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy drop >log/cifar100_IR50_basic_drop.log 2>&1 &
nohup python -u main.py --gpu 3 --imbanlance_rate 0.1 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy drop >log/cifar100_IR10_basic_drop.log 2>&1 &

nohup python -u main.py --gpu 0 --imbanlance_rate 0.005 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy pos_neg >log/cifar100_IR200_basic_pos_neg.log 2>&1 &
nohup python -u main.py --gpu 1 --imbanlance_rate 0.01 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy pos_neg >log/cifar100_IR100_basic_pos_neg.log 2>&1 &
nohup python -u main.py --gpu 2 --imbanlance_rate 0.02 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy pos_neg >log/cifar100_IR50_basic_pos_neg.log 2>&1 &
nohup python -u main.py --gpu 3 --imbanlance_rate 0.1 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy pos_neg >log/cifar100_IR10_basic_pos_neg.log 2>&1 &

nohup python -u main.py --gpu 0 --imbanlance_rate 0.005 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy neg_only >log/cifar100_IR200_basic_neg.log 2>&1 &
nohup python -u main.py --gpu 1 --imbanlance_rate 0.01 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy neg_only >log/cifar100_IR100_basic_neg.log 2>&1 &
nohup python -u main.py --gpu 2 --imbanlance_rate 0.02 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy neg_only >log/cifar100_IR50_basic_neg.log 2>&1 &
nohup python -u main.py --gpu 3 --imbanlance_rate 0.1 -a resnet32 --dataset cifar100 --num_classes 100 --lr 0.01 --loss_strategy neg_only >log/cifar100_IR10_basic_neg.log 2>&1 &

nohup python -u main.py -a resnext50 --dataset ImageNet-LT --num_classes 1000 --lr 0.1 --loss_strategy pos_neg >log/imagenet_basic_pos_neg.log 2>&1 &