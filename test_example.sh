if [[ "$1" == "" ]]; then
    echo "Error: Missing the dataset name"
    echo "./test_example.sh <dataset name> <number of cross fold (optional)>"
    exit
fi

# Test for inputed dataset

if [[ "$2" -ne "" ]]; then
    for ((i=0; i < $2; i++ ));
    do
        python -m tests.test_nusantara nusantara/nusa_datasets/$1/$1.py --subset_id "$1_fold$i"
    done
    exit
fi

python -m tests.test_nusantara nusantara/nusa_datasets/$1/$1.py
