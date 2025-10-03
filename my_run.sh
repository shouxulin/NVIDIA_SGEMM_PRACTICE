
kernel=8
dim=16
warmup=5
repeat=10
print=0
verify=0



dim_thres=8192

while [ $dim -le $dim_thres ]
do
    echo "################### Dim: ${dim} ###############################"
    for kernel in 0 7 8;
    do
        echo "Kernel: ${kernel}"
        cmd="./benchmark --kernel ${kernel} --dim ${dim} --warmup ${warmup} --repeat ${repeat} --print ${print} --verify ${verify}"
        echo $cmd
        eval $cmd
    done
    dim=$((dim * 2))
done