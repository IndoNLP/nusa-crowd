import pandas as pd
import urllib.request

def load_x_fact_dataset(dataset_path):
    columns = []
    data = []
    for i, line in enumerate(urllib.request.urlopen(dataset_path)):
        arr = line.decode('utf-8').strip().split('\t')
        if(i == 0):
            columns = arr
        else:
            data.append(arr)
    
    return_dataset = pd.DataFrame(data = data, columns = columns).reset_index()

    return return_dataset[return_dataset['language'] == 'id']
