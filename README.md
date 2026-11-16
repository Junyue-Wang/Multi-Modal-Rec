# MBD Multi-Modal Recommendation

A lightweight replication of the multi-modal recommendation experiments described in the [MBD paper](https://arxiv.org/abs/2409.17587). This project uses a sample of 3,000 customer IDs selected from the raw data. The processed customer features are saved to `data/processed/features.parquet`.

## Experiment steps

Run the notebooks in order:

1. **Data preparation** — Load the raw data, sample 3,000 customers, and prepare features.
2. **Unimodal models** — Train and evaluate a separate model for each modality.
3. **Blending** — Combine unimodal prediction probabilities using validation-set weights.
4. **Late fusion** — Combine representations learned independently from each modality, then train a classifier.
5. **Multi-modal (early fusion)** — Combine modalities before prediction and model their interactions with cross-attention.
6. **Evaluation** — Compare the models using the same evaluation metrics.

## Setup

Install dependencies and launch Jupyter:

```bash
pip install -r requirements.txt
jupyter lab
```

Start with `00_setup_and_data.ipynb` and follow the numbered notebooks in `notebooks/`. The processed feature file is `data/processed/features.parquet`.

## Note

This is a small-scale replication intended for experimentation. Its results may differ from the paper because of the 3,000-customer sample and differences in implementation and computing resources. Cite the MBD paper when using the dataset or reporting results.
