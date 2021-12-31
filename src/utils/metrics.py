import torch

def _sub_cost_f(a: str, b: str, default_cost: int):
    return 0 if a == b else default_cost

def char_error_rate(y_pred, y_true, del_cost: int=1, ins_cost: int=1, sub_cost: int=2):
    '''
    Min edit distance algorithm for computing the minimum edit distance btw two strings.
    The minimum edit distance is defined as the number of editing operations needed to transform
    one string into another.
    Operations are for ex: deletion, insertion, substitution

    y_pred: tensor of shape (S) of char predictions 
    y_true: tensor of shape (S) of char ground-truth 
    del_cost: cost for the deletion operation
    ins_cost: cost for the insertion operation
    sub_cost: cost for the substitution operation

    Is an instance of a dynamic programming algorithm, and compute the final min_edit_dist by
    constructing a matrix of distances. This matrix is constructed at each step based on the values
    computed in the previous step.
    '''

    n = y_pred.shape[0]
    m = y_true.shape[0]

    dist_m = torch.zeros((n+1, m+1), dtype=int)
    dist_m[0, 0] = 0

    # first row (src)
    for i in range(1, n+1):
        dist_m[i, 0] = dist_m[i-1, 0] + del_cost

    # first col (trg)
    for i in range(1, m+1):
        dist_m[0, i] = dist_m[0, i-1] + ins_cost

    # compute distance matrix bottom-up
    for i in range(1, n+1):
        for j in range(1, m+1):
            del_c = dist_m[i-1, j] + del_cost
            sub_c = dist_m[i-1, j-1] + _sub_cost_f(y_pred[i-1], y_true[j-1], sub_cost)
            ins_c = dist_m[i, j-1] + ins_cost
            dist_m[i, j] = min(del_c, sub_c, ins_c)

    return dist_m[n, m]

