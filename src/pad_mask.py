import torch
from torch.nn.utils.rnn import pad_sequence

    
def pad_and_mask(batch_obs,padding_value = 0):
    """
    Returns padded batch and the mask
    :param batch_obs: (list) the observation batch,a list of lists.
    
    :return padded: (torch.Tensor): padded batch
    :return mask: input mask

    Example
    input : 
        sample_1 = [
                    [1,2,3,4,5],
                    [2,4,5,6,8],
                    [3,2,1],
                    [1,2],
                    [9,8,7,6,5,4,3]
        ]
        sample_2 = [
                    [10,20,30,40],
                    [20,40,50,60],
                    [30,20,10],
                    [10,20]
        ]
        batch =  [sample_1,sample_2]
    output:
    padded_batch_obs:
            tensor([[[1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 1., 1., 0., 0.],
                [1., 1., 1., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0.],
                [1., 1., 1., 1., 1., 1., 1.]],

                [[1., 1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 1., 0., 0., 0.],
                [1., 1., 1., 0., 0., 0., 0.],
                [1., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0.]]])

    mask_obs:
            tensor([[[ 1.,  2.,  3.,  4.,  5.,  0.,  0.],
                [ 2.,  4.,  5.,  6.,  8.,  0.,  0.],
                [ 3.,  2.,  1.,  0.,  0.,  0.,  0.],
                [ 1.,  2.,  0.,  0.,  0.,  0.,  0.],
                [ 9.,  8.,  7.,  6.,  5.,  4.,  3.]],

                [[10., 20., 30., 40.,  0.,  0.,  0.],
                [20., 40., 50., 60.,  0.,  0.,  0.],
                [30., 20., 10.,  0.,  0.,  0.,  0.],
                [10., 20.,  0.,  0.,  0.,  0.,  0.],
                [ 0.,  0.,  0.,  0.,  0.,  0.,  0.]]])
    """
    padded_batch_obs = pad_(batch_obs)
    mask_obs = (padded_batch_obs != padding_value).any(dim=-1)
    return padded_batch_obs,(mask_obs == False)
    
def pad_(input):
    """
    performs the padding 
    
    """
    batch = [ [torch.tensor(obj) for obj in obs] for obs in input ]
    batch_obs = [pad_sequence(obs, batch_first=True, padding_value=0) for obs in batch]

    max_num_sequences = max(tensor.size(0) for tensor in batch_obs)
    max_length = 15 #15 # max(tensor.size(1) for tensor in batch_obs)

    padded_by_num_of_sequence = [ 
        torch.cat((tensor, torch.zeros(max_num_sequences - tensor.size(0), tensor.size(1))), dim = 0)
        for tensor in batch_obs]
    all_padded = [ 
        torch.cat((tensor, torch.zeros(tensor.size(0),max_length - tensor.size(1))), dim = 1)   
        for tensor in padded_by_num_of_sequence]
    
    return torch.stack(all_padded)

def to_mask_(input):
    """
    Reformat the observation to be masked
    """
    to_mask = []
    for obs in input:
        lengths = [len(item) for item in obs]
        ones = [[1]*length for length in lengths ]
        to_mask.append(ones)
    return to_mask
def pad_action_(input):
    """
    performs the padding of the action mask
    
    """
    b = [ torch.tensor(act).reshape(-1,1)  for act in input ]
    max_num_sequences = max(tensor.size(0) for tensor in b)
    padded_by_num_of_sequence = [ 
            torch.cat((tensor, torch.zeros(max_num_sequences - tensor.size(0), tensor.size(1))), dim = 0)
            for tensor in b]
    final = torch.stack(padded_by_num_of_sequence)
    return final
