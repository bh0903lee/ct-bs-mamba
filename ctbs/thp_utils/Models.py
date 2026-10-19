# Adapted from the Transformer Hawkes Process implementation
# (https://github.com/SimiaoZuo/Transformer-Hawkes-Process), MIT License; see THIRD_PARTY_NOTICES.md.
import torch
import Constants

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)
