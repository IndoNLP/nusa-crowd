Please name your PR after the issue it closes. You can use the following line: "Closes #ISSUE-NUMBER" where you replace the ISSUE-NUMBER with the one corresponding to your dataset.

### Checkbox
- [ ] Confirm that this PR is linked to the dataset issue.
- [ ] Create the dataloader script `nusantara/nusa_datasets/my_dataset/my_dataset.py` (please use only lowercase and underscore for dataset naming).
- [ ] Provide values for the `_CITATION`, `_DATASETNAME`, `_DESCRIPTION`, `_HOMEPAGE`, `_LICENSE`, `_URLs`, `_SUPPORTED_TASKS`, `_SOURCE_VERSION`, and `_NUSANTARA_VERSION` variables.
- [ ] Implement `_info()`, `_split_generators()` and `_generate_examples()` in dataloader script.
- [ ] Make sure that the `BUILDER_CONFIGS` class attribute is a list with at least one `NusantaraConfig` for the source schema and one for a nusantara schema.
- [ ] Confirm dataloader script works with `datasets.load_dataset` function.
- [ ] Confirm that your dataloader script passes the test suite run with `python -m tests.test_nusantara --path=nusantara/nusa_datasets/my_dataset/my_dataset.py`.
- [ ] If my dataset is local, I have provided an output of the unit-tests in the PR (please copy paste). This is OPTIONAL for public datasets, as we can test these without access to the data files.
