Please name your PR after the issue it closes. You can use the following line: "Closes #ISSUE-NUMBER" where you replace the ISSUE-NUMBER with the one corresponding to your dataset.

If the following information is NOT present in the issue, please populate:

- **Name:** *name of the dataset*
- **Dataset Description:** *short description of the dataset (or link to social media or blog post)*
- **Dataset URL:** *original URL of the dataset*
- **Dataset Publication:** *link to the dataset publication if available*
- **License:** *Type of license; please provide public for new datasets*
- **Task:** *The task(s) covered on the dataset*
- **Languages:** *list of languages included in the dataset. Put 3-digit ISO 639-3 code. (i.e. ind, sun, min, ...)*
- **Dialect/Style (optional):** *the dialect of the dataset (eg. Jawa ngapak, Bali dataran rendah, etc) if available*
- **Domain:** *the domain/source of the dataset? (eg. News, law, reviews, social media, etc) if available*
- **Dataset Size:** *the size of the dataset in terms of number of samples. If the dataset has pre-existing split, please report them individualy. (eg. 2000 train, 500 dev, 1000 test).*
- **Is Synthetic:** *Yes/No. Put yes if the dataset is generated synthetically somehow, for example by translating from other languages, or by generating from language models, or CFG, etc2.*

### Checkbox
- [ ] Confirm that this PR is linked to the dataset issue.
- [ ] Create the dataloader script `biodatasets/my_dataset/my_dataset.py` (please use only lowercase and underscore for dataset naming).
- [ ] Provide values for the `_CITATION`, `_DATASETNAME`, `_DESCRIPTION`, `_HOMEPAGE`, `_LICENSE`, `_URLs`, `_SUPPORTED_TASKS`, `_SOURCE_VERSION`, and `_BIGBIO_VERSION` variables.
- [ ] Implement `_info()`, `_split_generators()` and `_generate_examples()` in dataloader script.
- [ ] Make sure that the `BUILDER_CONFIGS` class attribute is a list with at least one `BigBioConfig` for the source schema and one for a bigbio schema.
- [ ] Confirm dataloader script works with `datasets.load_dataset` function.
- [ ] Confirm that your dataloader script passes the test suite run with `python -m tests.test_bigbio biodatasets/my_dataset/my_dataset.py`.
- [ ] If my dataset is local, I have provided an output of the unit-tests in the PR (please copy paste). This is OPTIONAL for public datasets, as we can test these without access to the data files.
