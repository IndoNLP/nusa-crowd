import pandas as pd
import urllib.request

def load_x_fact_dataset(dataset_url):
    columns = []
    data = []
    for i, line in enumerate(urllib.request.urlopen(dataset_url)):
        arr = line.decode('utf-8').strip().split('\t')
        if(i == 0):
            columns = arr
        else:
            data.append(arr)

    return pd.DataFrame(data = data, columns = columns)