# Contribution point guideline

To be considered as co-author, 10 contribution points is required.

> **Note**: The purpose of the point system is not to barrier collaboration, but to reward rare and high-quality dataset entries.
We might adjust the point requirement lower to accomodate more co-authorship, if needed.

## Implementing Data Loader

Implementing any data loader is granted +3 pts, unless otherwise specified on the Github issue.
More details [here](DATALOADER.md).

## NusaCatalogue's Datasheet Proposal

### Datasheet Proposal as the Dataset's Author(s)
Proposing a datasheet of a dataset is granted +2 pts.

As a support for data openness, for any data that is previously private, if Author(s) agree to make the dataset publicly available then additional +3 pts will be granted.

As a support for the development of local languages datasets:
- For dataset in Sundanese, Javanese, or Minangkabau, +2 pts will be granted
- For dataset in other local language, +3 pts will be granted

Based on our observation, we find that the common NLP tasks in Indonesian languages include: machine translation (MT), language modeling (LM), sentiment analysis (SA), and named entity recognition (NER). To encoureage more diverse NLP corpora, all other NLP tasks are considered rare and corresponding submission are eligible for the +2 contribution points. 

In addition, there are limited publicly available Indonesian NLP corpora involving other modality / multimodality, to encourage more coverage over these modalities, all submissions with these modalities will be eligible for a +2 contribution points

We understand that the quality of a dataset varies a lot. To support fairness in scoring datasets with different quality, for any dataset that does not achieve a certain minimum standard, 50% contribution score of the dataset will be penalized. This policy affects dataset that is collected with:
- Crawling without any manual validation process
- Machine / heuristic-rule labelled dataset, without any manual validation
- Machine translated dataset without any manual validation.

> **Note**: if there is more than 1 Author for a dataset, main Author will be eligible for nominating 1 more Author to be granted the same contribution score.

### Datasheet Proposal not as the Dataset's Author(s)
Proposing a datasheet of other people dataset is granted +1 pt.

## Listing Private Dataset
Each private dataset listed is granted +1 pt.
