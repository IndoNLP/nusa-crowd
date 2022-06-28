# Contribution point guideline

To be considered as co-author, 10 contribution points is required.

Note: The purpose of the point system is not to barrier collaboration, but to reward rare and high-quality dataset entries.
We might adjust the point requirement lower to accomodate more co-authorship, if needed.

## Data Loader

Implementing any data loader is granted +3 pts.
More detais[here](DATALOADER.md).

## Dataset Proposal

Point from dataset proposal depends on various factors:

### Size

We can have 4 different levels: Small, Medium, Large, XL

- Small (S): <1K (+1 pts)
- Medium (M): 1K<=x<10K (+ 2pts)
- Large (L): 10K<=x<1M (+3 pts)
- Extra Large (XL): >=1M (+4 pts)

### Task and Language Rarity

- Rare / No resource: No public dataset on this language / task. This dataset will be the new one for the particular language/task. (+6 pts)
- Uncommon: There are some resources on this local language, but they are very hard to find. (+ 3 pts)
- Common: Dataset for common task & language. (+1 pts)


### Quality (for labeled dataset)

- Excellent (E): High-quality dataset, eg. labelled/written/annotated **and** evaluated by humans with respectable annotator agreement. Annotation protocol is documented thoroughly in the paper. (pts x1.5)

- Good (G): eg. data is generated automatically (i.e. by crawling), but verified by human. Alternatively, data can be labelled by human with minimal/no verification. (pts x1)

- Poor (P): eg. data is fully machine-generated, with no verification. (pts x0.5)


## Examples

Let's assume a new sentiment analysis for one of Papuan language, consisting of 500 sentences.
For data size, it is considered small (+1 pts). While sentiment analysis is common, but the language itself is extremely rare and underrepresented, therefore we got +6 pts for this. Lastly, assuming the data is in high-quality, we'll obtain a total of (1 + 6) * 1.5 pts = 10.5pts, which is enough for authorship.

Another example, let's assume a new Natural Language Inference (NLI) dataset for Javanese. NLI by itself is not new for Indonesian languages, and Javanese resource is available. However, Javanese NLI is the first one even, hence it is still considered rare (+6 pts). Assuming the dataset is Small size, with Good quality, we end up with a total of 7 pts. By additionally, implementing the data loader for this dataset, we'll have a total of 10 pts, which is enough for authorship.
