Document Anonymization using Python, Diffusion Models, and Named Entity Recognition
==================================================================================

Anonymization of sensitive information in financial documents using, python, diffusion models and named entity recognition

Data is the fossil fuel of the machine learning world, essential for developing high quality models but in limited supply. 
Yet institutions handling sensitive documents — such as financial, medical, or legal records often cannot fully leverage their own data due to stringent privacy, compliance, and security requirements, making training high quality models difficult.

A promising solution is to replace the personally identifiable information (PII) with realistic synthetic stand-ins, whilst leaving the rest of the document in tact.

In this talk, we will discuss the use of open source tools and models that can be self hosted to anonymize documents. 
We will go over the various approaches for Named Entity Recognition (NER) to identify sensitive entities and the use of diffusion models to inpaint anonymized content.


Code used for talks at Europython 2025, Pycon Lithuania 2025.

Work based off [DiffUTE](https://arxiv.org/abs/2305.10825), [REPA-E](https://github.com/End2End-Diffusion/REPA-E), [diffusers](https://github.com/huggingface/diffusers), [spaCy](https://spacy.io/), 

## Installation