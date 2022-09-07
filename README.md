# A study on the effectiveness of AlphaFold's hidden MSA representation for protein mutation analysis 
## Thesis MSc in Data Science @ Bocconi University


## Virtual environments
To successfully run both OpenFold and DeepSequence, two separate conda environments are needed 


### **OpenFold conda environment**
As written in the README in the OpenFold directory, simply run:

```bash
openfold_model/scripts/install_third_party_dependencies.sh
```

This will create a conda environment named *openfold_venv*


### **DeepSequence conda environment**
Simply run:
```bash
conda env create --name deepsequence_pytorch_env -f deepsequence_model/environment.yml python==3.9.12
```

## Downloads

### **data_pickle/ folder**
Simply run:
```bash
# Inside the directory, download the data_pickle folder as tar.gz
wget https://openfold-deepsequence-integration.s3.eu-central-1.amazonaws.com/data_pickle.tar.gz

# Untar the folder
tar -zxvf data_pickle.tar.gz

# Remove the tar.gz
rm data_pickle.tar.gz
```

### **OpenFold weights**
Inside the directory, download OpenFold weights from [Huggingface](https://huggingface.co/nz/OpenFold/blob/main/finetuning_1.pt) <br>
Then create the params directory and move the weights into it with:
```bash
mkdir ./openfold_model/openfold/resources/openfold_params/
mv ./finetuning_1.pt ./openfold_model/openfold/resources/openfold_params/finetuning_1.pt
```

## Understanding the project
The best way to understand the integration between the two models is to understand the flow of data between them.

The steps to generate the data to train the model on are the following:
1. On [Google Colab](https://colab.research.google.com/github/aqlaboratory/openfold/blob/main/notebooks/OpenFold.ipynb), insert the reference sequence of the family and run the MSA. This sequence can also be found in the **get_openfold_predictions.ipynb notebook**. <br>
Once it is done, we can download the following variables from Colab:
    - deletion_matrix
    - msas <br>

    Few consideration:
    - The variables are saved as pickle in data_pickle/ and will be fed to OpenFold in the next step.
    - The data_pickle/ folder can be downloaded as explained at the beginning of the README.
    - The reason why the first part of the pipeline takes place on Colab instead of locally is to avoid downloading the biological databases.


2. Given the results of the previous step, now we can run **get_openfold_predictions.ipynb** to get the MSA representation. <br>
Few considerations:
    - Given the original MSA, OpenFold samples only 512 sequences to create the MSA representation, otherwise the computational complexity of the evoformer would make inference unfeasible. At each recycling step of OpenFold, 512 sequences are sampled again and affect the MSA representation. 
        - This is a problem because we want a representation of a fixed, and not mixed at each recycle, number of sequence, and we want to know which are those sequences. The solution is to set the recycle to 0 (so only one forward pass at inference) and to save the indices of the selected sequences from the MSA to create the MSA representation.
            - The section at the bottom of the README explains what scripts have been modified to change the behaviour of OpenFold.
        - Because 512 sequences is a small number of observation to train a neural network and it is not feasible to increase the sample rate directly with a normal GPU, the solution is to run the model multiple times and concatenate the results. 
        - In the data_pickle/ folder, you can find 3 folder, one for each variable saved when running the notebook: <br>
            - sel_seq/: contains the indices of the MSA that are sampled to make the MSA representation
            - not_sel_seq/: contains the indices that are not sampled. For each run, the length of the indices in sel_seq.pickle and not_sel_seq.pickle sum to the length of the sequences in the MSA
            - predict_result/: contains, among other things, the MSA representation
        - Future effort might be directed in re-introducing the recycling, but forcing to sample always the same sequences from the MSA. This would have the benefit of extracting even more processed representations, while being able to understand to which sequences those representation refer.
    - When running OpenFold, we also generate the MSA in 2d matrix form with shape (number_sequences, seq_len), where  values range from 0 to 19 for the amino acids and 21 means "-". This 2d matrix will be the input of the baseline model. Before being used, in the **deepsequence_experiments.ipynb** it will be one-hot encoded making it a 3d matrix of shape (number_sequences, seq_len, 20).


3. Once all data is ready, we can finally move on to running DeepSequence in the notebook **deepsequence_experiments.ipynb**.
Here we find:
    - Processing of the MSA from 2d to 3d one-hot as explained above
    - Processing of the MSA representation by stacking the results of 10 runs so that the datasets consists of 512*10=5120 MSA representation, even though some sequences might be sampled from the MSA multiple times.
    - Defined the DataHelperAugmented class, as an expansion of the DataHelper class from DeepSequence.
    - Training and evaluation of the models

## Modification to the OpenFold repository
1. config.py
    - "max_recycling_iters": 0 (from 3)

2. data_transform.py
    - Added print("\tCALLING MSA SAMPLING") inside the sample_msa() function
    - Added save_sampling_indices() function + call it inside the sample_msa() function

3. input_pipeline.py
    - Added print("\t\tRecycling from tensor") and print("\t\tRecycling from configs") in the process_tensors_from_config() function

4. get_openfold_predictions.ipynb
