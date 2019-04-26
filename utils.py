import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def tn_m_hidden(hidden, hidden_previous):
    h = []
    for hidden_i, hidden_v in enumerate(hidden):
        h.append(
            [
                (hidden_v[0]+hidden_previous[hidden_i][0])/2, 
                (hidden_v[1]+hidden_previous[hidden_i][1])/2
            ]
        ) 
    return h

def add_tn_params(parser):
    # ThinkNet params
    parser.add_argument('--tn_timesteps', type=int, default=1,
                        help='training ThinkNet timesteps')
    parser.add_argument('--tn_test_timesteps', type=int, default=10,
                        help='test ThinkNet timesteps')
    parser.add_argument('--tn_delta', action='store_true',
                        help='use Delta Loss')