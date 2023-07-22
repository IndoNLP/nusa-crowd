# Welcome to NusaCrowd!

<h3>143 datasets registered</h3>

![Dataset claimed](https://progress-bar.dev/83/?title=Datasets%20Claimed%20(119%20Datasets%20Claimed))

<!-- milestone starts -->
![Milestone 1](https://progress-bar.dev/100/?title=Milestone%201%20(30%20Datasets%20Completed))

![Milestone 2](https://progress-bar.dev/100/?title=Milestone%202%20(60%20Datasets%20Completed))

![Milestone 3](https://progress-bar.dev/100/?title=Milestone%203%20(100%20Datasets%20Completed))

![Milestone 4](https://progress-bar.dev/82/?title=Milestone%204%20(150%20Datasets%20Completed))
<!-- milestone ends -->

*Baca README ini dalam [Bahasa Indonesia](README.id.md).*

Indonesian NLP is underrepresented in research community, and one of the reasons is the lack of access to public datasets ([Aji et al., 2022](https://aclanthology.org/2022.acl-long.500/)). To address this issue, we initiate
**NusaCrowd**, a joint collaboration to collect NLP datasets for Indonesian languages. Help us collect and centralize Indonesian NLP datasets, and be a co-author of our upcoming paper.

## How to contribute?

You can contribute by proposing **unregistered NLP dataset** on [our record](https://indonlp.github.io/nusa-catalogue/). [Just fill out this form](https://forms.gle/31dMGZik25DPFYFd6), and we will check and approve your entry.

We will give **contribution points** based on several factors, including: **dataset quality**, **language scarcity**, or **task scarcity**.

You can also propose datasets from your past work that have not been released to the public.
In that case, you must first make your dataset open by uploading it publicly, i.e. via Github or Google Drive.

You can submit multiple entries, and if the total **contribution points** is already above the threshold, we will include you as a co-author (Generally it is enough to only propose 1-2 datasets). Read the full method of calculating points [here](POINTS.md).

> **Note**: We are not taking any ownership of the submitted dataset. See FAQ below.

## Any other way to help?

Yes! Aside from new dataset collection, we are also centralizing existing datasets in a single schema that makes it easier for researchers to use Indonesian NLP datasets. You can help us there by building dataset loader. More details about that [here](DATALOADER.md).

Alternatively, we're also listing NLP research papers of Indonesian languages where they do not open their dataset yet. We will contact the authors of these papers later to be involved in NusaCrowd. More about this is available in our [Slack server](https://join.slack.com/t/nusacrowd/shared_invite/zt-1bbvt4och-JkC7tzeL_eUk4UD6tl3kDg).

## FAQs

#### Who will be the owner of the submitted dataset?

NusaCrowd **do not** make a clone or copy of the submitted dataset. Therefore, the owner of any submitted dataset will remain to the original author. NusaCrowd simply build a dataloader, i.e. a file downloader + reader so simplify and standardize the data reading process. We also only collect and centralize metadata of the submitted dataset to be listed in [our catalogue](https://indonlp.github.io/nusa-catalogue/) for better discoverability of your dataset!
Citation to the original data owner is also provided for both NusaCrowd and in our catalogue.

#### How can I find the appropriate license for my dataset?

The license for a dataset is not always obvious. Here are some strategies to try in your search,

* check for files such as README or LICENSE that may be distributed with the dataset itself
* check the dataset webpage
* check publications that announce the release of the dataset
* check the website of the organization providing the dataset

If no official license is listed anywhere, but you find a webpage that describes general data usage policies for the dataset, you can fall back to providing that URL in the `_LICENSE` variable. If you can't find any license information, please note in your PR and put `_LICENSE="Unknown"` in your dataset script.

#### What if my dataset is not yet publicly available?

You can upload your dataset publicly first, eg. on Github.

#### Can I create a PR if I have an idea?

If you have an idea to improve or change the code of the nusa-crowd repository, please create an `issue` and ask for `feedback` before starting any PRs.

#### I am confused, can you help me?

Yes, you can ask for helps in NusaCrowd's community channel! Please join our [WhatsApp group](https://chat.whatsapp.com/Jn4nM6l3kSn3p4kJVESTwv) or [Slack server](https://join.slack.com/t/nusacrowd/shared_invite/zt-1bbvt4och-JkC7tzeL_eUk4UD6tl3kDg).


## Thank you!

We greatly appreciate your help!

The artifacts of this hackathon will be described in a forthcoming academic paper targeting a machine learning or NLP audience. Please refer to [this section](#contribution-guidelines) for your contribution rewards for helping Nusantara NLP. We recognize that some datasets require more effort than others, so please reach out if you have questions. Our goal is to be inclusive with credit!

## Citing NusaCrowd
```
@misc{cahyawijaya2022nusacrowd,
      title={NusaCrowd: Open Source Initiative for Indonesian NLP Resources}, 
      author={Samuel Cahyawijaya and Holy Lovenia and Alham Fikri Aji and Genta Indra Winata and Bryan Wilie and Rahmad Mahendra and Christian Wibisono and Ade Romadhony and Karissa Vincentio and Fajri Koto and Jennifer Santoso and David Moeljadi and Cahya Wirawan and Frederikus Hudi and Ivan Halim Parmonangan and Ika Alfina and Muhammad Satrio Wicaksono and Ilham Firdausi Putra and Samsul Rahmadani and Yulianti Oenang and Ali Akbar Septiandri and James Jaya and Kaustubh D. Dhole and Arie Ardiyanti Suryani and Rifki Afina Putri and Dan Su and Keith Stevens and Made Nindyatama Nityasya and Muhammad Farid Adilazuarda and Ryan Ignatius and Ryandito Diandaru and Tiezheng Yu and Vito Ghifari and Wenliang Dai and Yan Xu and Dyah Damapuspita and Cuk Tho and Ichwanul Muslim Karo Karo and Tirana Noor Fatyanosa and Ziwei Ji and Pascale Fung and Graham Neubig and Timothy Baldwin and Sebastian Ruder and Herry Sujaini and Sakriani Sakti and Ayu Purwarianti},
      year={2022},
      eprint={2212.09648},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

<!--
## Acknowledgements

This hackathon guide was heavily inspired by [the BigScience Datasets Hackathon](https://github.com/bigscience-workshop/data_tooling/wiki/datasets-hackathon).
 -->
