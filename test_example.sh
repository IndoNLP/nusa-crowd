if [[ "$1" == "" ]]; then
    echo "Error: Missing the dataset name"
    echo "./test_example.sh <dataset name>"
    exit
fi

# Test for inputed dataset
python -m tests.test_nusantara nusantara/nusa_datasets/$1/$1.py
