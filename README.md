<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Meta-HAR</h3>

  <p align="center">
    Meta-HAR: Federated Representation Learning for Human Activity Recognition
    <br />
    <a href="https://dl.acm.org/doi/pdf/10.1145/3442381.3450006"><strong>Paper published in TheWebConf 2021 </strong></a>
    <br />
  </p>

<!-- GETTING STARTED -->
### Dataset

Details in ./data_process/readme.md 

1. For collected dataset: 
    ```sh
    cd data_process
    python feature_extraction.py --in_dir 'dir stores the original txt data' --out_dir 'dir whic is used to store pickle data'   
    ```
    The ``feature_extraction.py`` generates pickle files and the ``trans_dict_collect.pickle`` file. 

2. For processing of the HHAR dataset please refer to: https://github.com/yscacaca/HHAR-Data-Process. 
   To run on public dataset for yourself, make the dataset to have the same format as mentioned in the ./data_process/readme.md

<!-- Run -->
## Run
1. data process as mentioned above.
2. Run Meta-HAR with default hyper-parameters. 
```sh 
    python Central.py   # for central model. 
    python meta-har.py  # for meta-har
```
Note: Configure your own data and output dirs.  

## Others
To run other baselines:
1. Reptile: Change the ``norm_embed`` to ``norm_cce`` in the Meta-HAR and remove fine-tune. 
2. Meta-HAR-CE: Use "target" instead of "target_t" in fine-tune.

<!-- CONTACT -->
## Contact

Chenglin Li - ch11@ualberta.ca


