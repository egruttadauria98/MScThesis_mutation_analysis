import numpy as np
import torch


def compute_amino_indices_from_tensor(t):

    indeces_amino = []

    for row in t:
        indeces_amino.append(torch.argmax(row).item())

    indeces_amino = np.array(indeces_amino)

    return indeces_amino


def compute_sequence_accuracy(input_seq, output_seq, slice=None):

    if slice is not None:
        input_seq = input_seq[slice]
        output_seq = output_seq[slice]

    return (sum(input_seq == output_seq)/len(input_seq))*100


def reconstruction_accuracy_per_aminoacid(model, data_helper, n_samples=10, finetune=False):

    '''
    This function test the accuracy of a model by comparing the ratio of aminoacids correctly reconstructed by the VAE.

    The goal is not to achieve a 100% accuracy because the VAE samples from the latent dimension and outputs a probability distribution over the aminoacid sequence,
    but to understand the extent to which the trained model will reconstruct the input sequence.
    '''

    # Accuracy on the full sequence
    accuracy_full_sequence = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    if finetune:
        batch_order = np.arange(data_helper.x_train_finetune.shape[0])
    else:
        batch_order = np.arange(data_helper.x_train.shape[0])

    batch_index = np.random.choice(batch_order, n_samples, p=None).tolist()

    for index in batch_index:

        data_batch = torch.Tensor(data_helper.x_train[index]).to(device)

        # Compute amino acid indices for the sequence from the one-hot encoding of the input
        input_index_amino = compute_amino_indices_from_tensor(t=data_batch)

        if finetune:
            data_batch = torch.Tensor(data_helper.x_train_finetune[index]).to(device)

        # Extract only reconstructed_x and reshape it to 2D tensor
        output, _, _ = model(data_batch)
        output = output.reshape(data_helper.seq_len, data_helper.alphabet_size)

        output_index_amino = compute_amino_indices_from_tensor(t=output)

        accuracy_full_sequence += compute_sequence_accuracy(input_index_amino, output_index_amino)
        #accuracy_restricted_sequence += compute_sequence_accuracy(input_index_amino, output_index_amino, slice=)

    return accuracy_full_sequence/n_samples