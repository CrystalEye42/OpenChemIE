# Data and sample scripts for the case study on FDA approved drug components (as described in the manuscript)

## Quick start

First deploy your ASKCOS instance following the instruction at [ASKCOS wiki](https://askcos-docs.mit.edu/guide/1-Introduction/1.1-Introduction.html). Then under the folder for the case study, run the python scripts to send the tree building queries as described in the SI of the manuscript, e.g.,

```shell
$ cd case_study_for_FDA_approved_drugs
$ python send_queries_for_baseline.py
$ python send_queries_for_rerun_1.py
$ python send_queries_for_rerun_2.py
$ python send_queries_for_rerun_3.py
```

These would queue up asynchronous queries to be executed one by one (or few by few) and the results will be available for viewing as they become ready at the frontend under your account. The only required python dependency is the `requests` library (pip installable), and you need to change the `HOST`, `PORT`, `USERNAME`, and `PASSWORD` in the scripts before sending, if different from the default.

## Details on the data files and optional preprocessing

- `compilation_of_cder_nme_and_new_biologic_approvals_1985-2023.csv` 

This is the raw data dumped from the [FDA website](https://www.fda.gov/drugs/development-approval-process-drugs/novel-drug-approvals-fda), accessed on Nov 2024 and upon clicking *Compilation of CDER New Molecular Entity (NME) Drug and New Biologic Approvals*.

- `FDA_approved_filtered_19-23.csv`

This is the processed data file used for the tree building queries, containing the list of target names and SMILES. It can be reproduced by

```shell
$ cd case_study_for_FDA_approved_drugs
$ python preprocess.py
```

The only required python dependencies are `requests`, `rdkit`, and `tqdm` (all pip installable), and you need to change the `HOST` and `PORT` in the script before sending, if different from the default. There may be some minor variations in the processed file, subject to the behavior of the name resolver API of NIH which is external and beyond our control.
