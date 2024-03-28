#!/bin/bash
if [ "$1" == "gxx" ]; then
    if [ "$2" == "amazon" ]; then
        ./CPUTucker-gxx -i ~/amazon-reviews.tns -o 3 -r 10 >> output10.txt
        ./CPUTucker-gxx -i ~/amazon-reviews.tns -o 3 -r 20 >> output20.txt
        ./CPUTucker-gxx -i ~/amazon-reviews.tns -o 3 -r 30 >> output30.txt
    elif [ "$2" == "nell" ]; then
        ./CPUTucker-gxx -i ~/nell-2.tns -o 3
    else
        echo "Invalid dataset. Please provide either 'amazon' or 'nell' as a second argument."
    fi
elif [ "$1" == "avx" ]; then
    if [ "$2" == "amazon" ]; then
        ./CPUTucker-avx -i ~/amazon-reviews.tns -o 3
    elif [ "$2" == "nell" ]; then
        ./CPUTucker-avx -i ~/nell-2.tns -o 3
    else
        echo "Invalid dataset. Please provide either 'amazon' or 'nell' as a second argument."
    fi
else
    echo "Invalid command. Please provide either 'gxx' or 'avx' as an argument."
fi