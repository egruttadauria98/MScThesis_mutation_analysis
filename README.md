# OpenFold-DeepSequence integration for protein 

The best way to understand the integration between the two models is to understand the flow of data between them.

First of all, DeepSequence is trained on a single protein family at a time and the first step is to generate a MSA which can be manipulated by both models. <br>
Given that we are training a Neural Network, the data in the data_pickle/ folder relates to the BLAT_ECOLX family because it has one of the biggest MSA and therefore more training examples.

The steps to generate the data to train the model on are the following:
1. On [Google Colab](https://colab.research.google.com/github/aqlaboratory/openfold/blob/main/notebooks/OpenFold.ipynb), insert the reference sequence of the family and run the MSA. This sequence can also be found in the get_openfold_predictions.ipynb notebook. <br>
Once it is done, we can download the following variables from Colab:
    - deletion_matrix
    - msas <br>

    Few consideration:
    - The variables are saved as pickle in data_pickle/ and will be fed to OpenFold in the next script.
    - The reason why the first part of the pipeline takes place on Colab instead of locally is to avoid to download the biological databases.


2. Given the results of the previous step, now we can run **get_openfold_predictions.ipynb** to get the MSA representation. <br>
Few considerations:
    - Given the original MSA, OpenFold samples only 512 sequences to create the MSA representation, otherwise the computational complexity of the evoformer would make inference unfeasible. At each recycling step of OpenFold, 512 sequences are sampled again and they will affect the MSA representation. 
        - This is a problem because we want a representation of a fixed, and not mixed at each recycle, number of sequence, and we want to know which are those sequences. The solution is to set the recycle to 0 (so only one forward pass at inference) and to save the indices of the selected sequences from the MSA to create the MSA representation.
        - Because 512 sequences is a small number of observation to train a neural network and it is not feasible to increase the sample rate directly with a normal GPU, the solution is to run the model multiple times and concatenate the results. 
        - In the data_pickle/ folder, you can find 3 folder, one for each variable saved when running the notebook: <br>
            - sel_seq/: contains the indices of the MSA that are sampled to make the MSA representation
            - not_sel_seq/: contains the indices that are not sampled. For each run, the length of the indices in sel_seq.pickle and not_sel_seq.pickle sum to the length of the sequences in the MSA
            - predict_result/: contains, among other things, the MSA representation
        - Future effort might be directed in re-introducing the recycling, but forcing to sample always the same sequences from the MSA. This would have the benefit of extracting even more processed representations, while being able to understand to which sequences those representation refer.
    - When running OpenFold, we also generate the MSA in 2d matrix form with shape (number_sequences, seq_len), where  values range from 0 to 19 for the amino acids and 21 means "-". This 2d matrix will be the input of the baseline model. Before being used, in the deepsequence_experiments.ipynb it will be one-hot encoded making it a 3d matrix of shape (number_sequences, seq_len, 20).


3. Once all data is ready, we can finally move on to running DeepSequence in the notebook **deepsequence_experiments.ipynb**.
Here we find:
    - Processing of the MSA from 2d to 3d one-hot as explained above
    - Processing of the MSA representation by stacking the results of 10 runs so that the datasets consists of 512*10=5120 MSA representation, even though some sequences might be sampled from the MSA multiple times.
    - Defined the DataHelperAugmented class, as an expansion of the DataHelper class from DeepSequence.
    - Training and evaluation of the models

### Modification in the OpenFold repository
1. config.py
    - "max_recycling_iters": 0 (from 3)

2. data_transform.py
    - Added print("\tCALLING MSA SAMPLING") inside the sample_msa() function
    - Added save_sampling_indices() function + call it inside the sample_msa() function

3. input_pipeline.py
    - Added print("\t\tRecycling from tensor") and print("\t\tRecycling from configs") in the process_tensors_from_config() function

4. Added get_openfold_predictions.ipynb
