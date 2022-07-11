from datasets import load_dataset

if __name__ == "__main__":
    # data = load_dataset("nusantara/nusa_datasets/smsa/smsa.py", name='smsa_source')
    # print(data['train'][0])

    # data = load_dataset("nusantara/nusa_datasets/id_hatespeech/id_hatespeech.py", name='id_hatespeech_source')
    # print(data['train'][0])

    # data = load_dataset("nusantara/nusa_datasets/nusax_senti/nusax_senti.py", name='nusax_senti_ace_source')
    print("\n\n\nSpecific language")
    data = load_dataset("nusantara/nusa_datasets/nusax_senti/nusax_senti.py", name='nusax_senti_ind_source')
    print(data['train'][0])

    print("\n\n\nAll language")
    data = load_dataset("nusantara/nusa_datasets/nusax_senti/nusax_senti.py", name='nusax_senti_source')
    print(data['train'][0])

    # data = load_dataset("nusantara/nusa_datasets/nusax_senti/nusax_senti.py", name='nusax_senti_min_source', split='validation')
    # print(data[0])

    # data = load_dataset("nusantara/nusa_datasets/nusax_senti/nusax_senti.py", name='nusax_senti_ind_source', split='test')
    # print(data[0])

    # python -m tests.test_nusantara nusantara/nusa_datasets/nusax_senti/nusax_senti.py
    # make check_file=nusantara/nusa_datasets/nusax_senti/nusax_senti.py