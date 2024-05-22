import copy
import random

# typing 
from typing import List, Tuple
import time
import torch

# TODO
# from transformers import LlamaTokenizer
# tokenizer=LlamaTokenizer.from_pretrained("/home/lyh/weights/hf/vicuna_v13/7B/")

TOPK = 10  # topk for sparse tree

from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)


def timer(func):
    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        result = func(*args, **kwargs)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f'{func.__name__} took {elapsed} seconds')
        return result

    return wrapper


def prepare_logits_processor(
        temperature: float = 0.0,
        repetition_penalty: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    if temperature > 1e-5:
        if temperature >= 1e-5 and temperature != 1.0:
            processor_list.append(TemperatureLogitsWarper(temperature))
        if repetition_penalty > 1.0:
            processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
        if 1e-8 <= top_p < 1.0:
            processor_list.append(TopPLogitsWarper(top_p))
        if top_k > 0:
            processor_list.append(TopKLogitsWarper(top_k))
        return processor_list


# test_processor = prepare_logits_processor(
#         0.0, 0.0, -1, 1
#     )


def pad_path(path: List[int], length: int, pad_value: int = -2) -> List[int]:
    """
    Pad the given path list with a specific value up to a specified length.

    Parameters:
    - path (list): The original list that needs padding.
    - length (int): The desired length of the padded list.
    - pad_value (optional, default=-2): The value to use for padding.

    Returns:
    - list: A new list based on the original path but padded to the desired length.

    Example:
    >>> pad_path([1,2,3], 5)
    [1, 2, 3, -2, -2]

    Note:
    If the given path is already longer than the specified length,
    then no padding occurs, and the original path is returned.
    """

    # Calculate the number of padding values needed by subtracting the length
    # of the path from the desired length.
    # Append the padding values to the original path and return the new list.
    return path + [pad_value] * (length - len(path))


def generate_tree_buffers(tree_choices, device="cuda"):
    TOPK = max([max(x) for x in tree_choices]) + 1
    sorted_tree_choices = sorted(tree_choices, key=lambda x: (len(x), x))
    tree_len = len(sorted_tree_choices) + 1

    # Initialize depth_counts to keep track of how many choices have a particular depth
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    print ("depth_counts:",depth_counts)

    tree_attn_mask = torch.eye(tree_len, tree_len) 
    tree_attn_mask[:, 0] = 1
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            ancestor_idx = []
            # retrieve ancestor position
            if len(cur_tree_choice) == 1:
                continue
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]) + 1)
            tree_attn_mask[j + start + 1, ancestor_idx] = 1
        start += depth_counts[i]
    print ("tree_attn_mask:",tree_attn_mask)

    tree_indices = torch.zeros(tree_len, dtype=torch.long)
    p_indices = [0 for _ in range(tree_len - 1)]
    b_indices = [[] for _ in range(tree_len - 1)]
    tree_indices[0] = 0
    start = 0
    bias = 0
    for i in range(len(depth_counts)):
        inlayer_bias = 0
        b = []
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            cur_parent = cur_tree_choice[:-1]
            if j != 0:
                if cur_parent != parent:
                    bias += 1
                    inlayer_bias += 1
                    parent = cur_parent
                    b = []
            else:
                parent = cur_parent
            tree_indices[start + j + 1] = cur_tree_choice[-1] + TOPK * (i + bias) + 1
            p_indices[start + j] = inlayer_bias
            if len(b) > 0:
                b_indices[start + j] = copy.deepcopy(b)
            else:
                b_indices[start + j] = []
            b.append(cur_tree_choice[-1] + TOPK * (i + bias) + 1)
        start += depth_counts[i]

    p_indices = [-1] + p_indices
    tree_position_ids = torch.zeros(tree_len, dtype=torch.long)
    start = 0
    for i in range(len(depth_counts)):
        tree_position_ids[start + 1: start + depth_counts[i] + 1] = i + 1
        start += depth_counts[i]

    retrieve_indices_nest = []
    retrieve_paths = []
    for i in range(len(sorted_tree_choices)):
        cur_tree_choice = sorted_tree_choices[-i - 1]
        retrieve_indice = []
        if cur_tree_choice in retrieve_paths:
            continue
        else:
            for c in range(len(cur_tree_choice)):
                retrieve_indice.append(sorted_tree_choices.index(cur_tree_choice[:c + 1]))
                retrieve_paths.append(cur_tree_choice[:c + 1])
        retrieve_indices_nest.append(retrieve_indice)
    max_length = max([len(x) for x in retrieve_indices_nest])
    retrieve_indices = [pad_path(path, max_length) for path in retrieve_indices_nest]
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
    retrieve_indices = retrieve_indices + 1
    retrieve_indices = torch.cat([torch.zeros((retrieve_indices.shape[0], 1), dtype=torch.long), retrieve_indices],
                                 dim=1)

    maxitem = retrieve_indices.max().item() + 5

    def custom_sort(lst):
        # sort_keys=[len(list)]
        sort_keys = []
        for i in range(len(lst)):
            sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
        return sort_keys

    retrieve_indices = retrieve_indices.tolist()
    retrieve_indices = sorted(retrieve_indices, key=custom_sort)
    retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)

    p_indices = torch.tensor(p_indices)
    p_indices_new = p_indices[retrieve_indices]
    p_indices_new = p_indices_new.tolist()

    b_indices = [[]] + b_indices
    b_indices_new = []
    for ib in range(retrieve_indices.shape[0]):
        iblist = []
        for jb in range(retrieve_indices.shape[1]):
            index = retrieve_indices[ib, jb]
            if index == -1:
                iblist.append([])
            else:
                b = b_indices[index]
                if len(b) > 0:
                    bt = []
                    for bi in b:
                        bt.append(torch.where(tree_indices == bi)[0].item())
                    iblist.append(torch.tensor(bt, device=device))
                else:
                    iblist.append(b)
        b_indices_new.append(iblist)

    # Aggregate the generated buffers into a dictionary
    tree_buffers = {
        "tree_attn_mask": tree_attn_mask.unsqueeze(0).unsqueeze(0),
        "tree_indices": tree_indices,
        "tree_position_ids": tree_position_ids,
        "retrieve_indices": retrieve_indices,
    }

    # Move the tensors in the dictionary to the specified device
    tree_buffers = {
        k: v.clone().to(device)
        if isinstance(v, torch.Tensor)
        else torch.tensor(v, device=device)
        for k, v in tree_buffers.items()
    }
    tree_buffers["p_indices"] = p_indices_new
    tree_buffers["b_indices"] = b_indices_new
    return tree_buffers


def initialize_tree(input_ids, model, tree_attn_mask, past_key_values, logits_processor,mode=None):
    tree_logits, outputs, logits, hidden_state, sample_token = model(
        input_ids, past_key_values=past_key_values, output_orig=True, logits_processor=logits_processor, 
        mode=mode,
    )
    model.base_model.model.tree_mask = tree_attn_mask
    return tree_logits, logits, hidden_state, sample_token


def reset_tree_mode(
        model,
):
    model.base_model.model.tree_mask = None
    model.base_model.model.tree_mode = None


def reset_past_key_values(passed_key_values: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Resets the current lengths in the passed key-values to zero.

    This function is designed to be used during the evaluation of a baseline model.
    It iterates through each layer's key-values and sets their current lengths to zero,
    effectively resetting their state.

    Args:
    - passed_key_values (list of torch.Tensor): Contains past hidden states and past attention values for each layer.

    Returns:
    - passed_key_values (list of torch.Tensor): Updated past hidden states and past attention values with reset lengths.
    """
    for i in range(len(passed_key_values)):
        for j in range(2):
            passed_key_values[i][j].current_length.fill_(0)
    return passed_key_values


def generate_candidates(tree_logits, tree_indices, retrieve_indices, sample_token, logits_processor):
    sample_token = sample_token.to(tree_indices.device)

    candidates_logit = sample_token[0]

    candidates_tree_logits = tree_logits[0]

    candidates = torch.cat([candidates_logit, candidates_tree_logits.view(-1)], dim=-1)

    tree_candidates = candidates[tree_indices]

    tree_candidates_ext = torch.cat(
        [tree_candidates, torch.zeros((1), dtype=torch.long, device=tree_candidates.device) - 1], dim=0)

    cart_candidates = tree_candidates_ext[retrieve_indices]

    if logits_processor is not None:
        candidates_tree_prob = tree_logits[1]
        candidates_prob = torch.cat(
            [torch.ones(1, device=candidates_tree_prob.device, dtype=torch.float32), candidates_tree_prob.view(-1)],
            dim=-1)

        tree_candidates_prob = candidates_prob[tree_indices]
        tree_candidates_prob_ext = torch.cat(
            [tree_candidates_prob, torch.ones((1), dtype=torch.float32, device=tree_candidates_prob.device)], dim=0)
        cart_candidates_prob = tree_candidates_prob_ext[retrieve_indices]
    else:
        cart_candidates_prob = None
    # Unsqueeze the tree candidates for dimension consistency.
    tree_candidates = tree_candidates.unsqueeze(0)
    return cart_candidates, cart_candidates_prob, tree_candidates


def tree_decoding(
        model,
        tree_candidates,
        past_key_values,
        tree_position_ids,
        input_ids,
        retrieve_indices,
):
    position_ids = tree_position_ids + input_ids.shape[1]

    outputs, tree_logits, hidden_state = model(
        tree_candidates,
        output_orig=True,
        past_key_values=past_key_values,
        position_ids=position_ids,
        init=False,
    )

    logits = tree_logits[0, retrieve_indices]
    return logits, hidden_state, outputs


def evaluate_posterior(
        logits: torch.Tensor,
        candidates: torch.Tensor,
        logits_processor,
        cart_candidates_prob,
        op,
        p_indices,
        tree_candidates,
        b_indices,
        mode=None,
) -> Tuple[torch.Tensor, int]:
    """
    Evaluate the posterior probabilities of the candidates based on the provided logits and choose the best candidate.

    Depending on the temperature value, the function either uses greedy decoding or evaluates posterior
    probabilities to select the best candidate.

    Args:
    - logits (torch.Tensor): Predicted logits of shape (batch_size, sequence_length, vocab_size).
    - candidates (torch.Tensor): Candidate token sequences.
    - temperature (float): Softmax temperature for probability scaling. A value of 0 indicates greedy decoding.
    - posterior_threshold (float): Threshold for posterior probability.
    - posterior_alpha (float): Scaling factor for the threshold.

    Returns:
    - best_candidate (torch.Tensor): Index of the chosen best candidate.
    - accept_length (int): Length of the accepted candidate sequence.
    """
    # Greedy decoding based on temperature value
    if logits_processor is None:
        # Find the tokens that match the maximum logits for each position in the sequence
        posterior_mask = (
                candidates[:, 1:].to(logits.device) == torch.argmax(logits[:, :-1], dim=-1)
        ).int()
        candidates_accept_length = (torch.cumprod(posterior_mask, dim=1)).sum(dim=1)
        accept_length = candidates_accept_length.max()
        # Choose the best candidate
        if accept_length == 0:
            # Default to the first candidate if none are accepted
            best_candidate = torch.tensor(0, dtype=torch.long, device=candidates.device)
        else:
            best_candidate = torch.argmax(candidates_accept_length).to(torch.long)
        return best_candidate, accept_length, logits[best_candidate, accept_length]

    else:
        cart_candidates_prob = cart_candidates_prob.to(logits.device)
        accept_length = 1
        accept_cand = candidates[0][:1]
        best_candidate = 0
        for i in range(1, candidates.shape[1]):
            if i != accept_length:
                break
            adjustflag = False
            is_eq = (candidates[:, :accept_length] == accept_cand).all(dim=1)
            fi = torch.nonzero(is_eq, as_tuple=True)[0][0]
            gt_logits = logits[fi, i - 1][None]
            gt_logits = logits_processor(None, gt_logits)[0]
            gtp = torch.softmax(gt_logits, dim=0)
            candidates_set = []

            if mode == 'spechub':
                """two drafts:    
                        We verify sampling the pair of top-1 and any other index. 
                        For example, if 'a' is the token with highest probability, we sample the pair (a,i) where i is any other token
                        with Q(i,a) = q(i) and Q(a,i) = q(a) * q(i) / (1-q(a))
                        
                        For verification, give target distribution p, and sample (a,i) or (i,a) ~ Q, 
                        if i,a: 
                            accept i with min (1,p(i) / Q(i,a)) # accept as much i as first draft as possible
                        residual p'(i) = p(i) - Q(i,a) * min (1,p(i) / Q(i,a)) # leftover p(i) that is not captured by first draft
                        residual Q'(i,a) = Q(i,a) - Q(i,a) * min (1,p(i) / Q(i,a))
                        if a,i:
                            accept i with min(1, p'(i) / Q(a,i)) # accept i as second draft
                        residual p''(i) = p'(i) - Q(a,i) * min(1, p'(i) / Q(a,i)) # the p(i) that is not captured . 
                        residual Q'(a,i) = Q(a,i) - Q(a,i) * min(1, p'(i) / Q(a,i))

                        if a,i:
                            accept a with min(1, p(a)/\sum_i Q'(a,i))
                            p'(a) = p(a) - \sum_i Q'(a,i) * min(1, p(a) / \sum_i Q'(a,i))
                        if i,a: 
                            accept a with min(1, p'(a) / \sum_i Q'(i,a))
                        
                        return residual p''(a) = p'(a) - \sum_i Q'(i,a) * min(1, p'(a) / \sum_i Q'(i,a))"""
                two_cands = []
                for j in range(candidates.shape[0]):
                    if is_eq[j]:
                        x = candidates[j, i]
                        xi = x.item()
                        if xi in candidates_set or xi == -1:
                            continue
                        candidates_set.append(xi)
                        two_cands.append((candidates[j, i], j))
                if len(two_cands) == 1:
                    r = random.random()
                    x = two_cands[0][0]
                    px = gtp[xi]
                    qx = cart_candidates_prob[two_cands[0][1], i]
                    if qx <= 0:
                        continue
                    acp = px / qx
                    if r <= acp:
                        accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                        accept_length += 1
                        best_candidate = two_cands[0][1]
                    else:
                        q = op[i - 1][p_indices[two_cands[0][1]][i]].clone()
                        b = b_indices[two_cands[0][1]][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            q[mask] = 0
                            q = q / q.sum()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
                elif len(two_cands) == 2:
                    x1 = two_cands[0][0]
                    x2 = two_cands[1][0]
                    
                    q = op[i - 1][p_indices[two_cands[0][1]][i]].clone() # shape (vocab_size)
                    a = torch.argmax(q) # shape (1)
                    # to avoid numerical issue, if p[a] > 1-1e-4, we directly accept a
                    if gtp[a] > 1- 1e-4:
                        accept_cand = torch.cat((accept_cand, a[None]), dim=0)
                        accept_length += 1
                        best_candidate = two_cands[0][1] if x1 == a else two_cands[1][1]
                        # print ('directly accept a',a.item())
                        continue
                    # print ("x1",x1.item(),"x2",x2, 'a', a)
                    # print ('q[a]',q[a].item(),'q[x1]',q[x1.item()].item(),'q[x2]',q[x2.item()].item())
                    # print ("gtp[x1]",gtp[x1.item()].item(),"gtp[x2]",gtp[x2.item()].item(),"gtp[a]",gtp[a].item())
                    def residual(p,q, a):
                        pp = torch.max(torch.zeros_like(p,device=p.device), p - q)
                        pp[a] = p[a]
                        qp = torch.max(torch.zeros_like(q, device=q.device), q - p)
                        qp[a] = q[a]
                        # print ('pp sum',pp.sum(),"qp sum",qp.sum())
                        return pp, qp
                    
                        
                    if x2 == a:
                        px1 = gtp[x1.item()]
                        # q_ia = q(i)
                        qx1 = cart_candidates_prob[two_cands[0][1], i] 
                        acp = px1 / qx1
                        r = random.random()
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x1[None]), dim=0)
                            accept_length += 1
                            best_candidate = two_cands[0][1]
                            continue
                    gtp, q_ia = residual(gtp, q, a)
                    q_ia[a] = 0
                    # print ("after ia i, gtp[x1]",gtp[x1.item()],"gtp[x2]",gtp[x2.item()])
                    # print ('q_ia sum',q_ia.sum())
                    def get_q_ai(q,a):
                        """
                        use log and softmax to avoid numerical instability
                        """
                        logq = torch.log(q+5e-5)
                        logq[a] = - torch.inf
                        q_normalized = torch.softmax(logq, dim=0)
                        return q[a]*q_normalized
                    q_ai = get_q_ai(q,a)
                    if x1 == a:
                        px2 = gtp[x2.item()]
                        # q_ai = q(a) * q(i) / (1-q(a))
                        qx2 = q_ai[x2.item()]
                        acp = px2 / qx2
                        r = random.random()
                        if r <= acp or qx2.isnan():
                            accept_cand = torch.cat((accept_cand, x2[None]), dim=0)
                            accept_length += 1
                            best_candidate = two_cands[1][1]
                            continue
                    # print ('q_ai sum',q_ai.sum(), 'q[a]', q[a].item())
                    gtp, q_ai = residual(gtp, q_ai, a)
                    # print ("after ai i, gtp[x1]",gtp[x1.item()],"gtp[x2]",gtp[x2.item()])
                    # print ('q_ai sum',q_ai.sum())
                    
                    if x1 == a:
                        pa = gtp[a]
                        acp = pa / (q_ai.sum())
                        r = random.random()
                        # print ('acp', acp, 'ai a: r', r)
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, a[None]), dim=0)
                            accept_length += 1
                            best_candidate = two_cands[0][1]
                            continue
                    gtp[a] = max(gtp[a] - q_ai.sum(), 0)
                    # print ("after ai a, gtp[x1]",gtp[x1.item()],"gtp[x2]",gtp[x2.item()])
                    if x2 == a:
                        pa = gtp[a]
                        acp = pa / q_ia.sum()
                        r = random.random()
                        if r <= acp or q_ia.sum() == 0 or q_ia.sum().isnan():
                            accept_cand = torch.cat((accept_cand, a[None]), dim=0)
                            accept_length += 1
                            best_candidate = two_cands[1][1]
                            continue
                    gtp[a] = max(gtp[a] - q_ia.sum(), 0)
                    # print ("after ia a, gtp[x1]",gtp[x1.item()],"gtp[x2]",gtp[x2.item()])
                    # print ("gtp.sum()",gtp.sum())
                    gtp = gtp / gtp.sum()
                    adjustflag = True
                else:
                    raise ValueError(f'spechub only supports two candidates, got {len(two_cands)} candidates.')
            elif mode == 'RRS_wo_replacement' or mode == 'RRS':

                for j in range(candidates.shape[0]):
                    if is_eq[j]:
                        x = candidates[j, i]
                        xi = x.item()
                        if xi in candidates_set or xi == -1:
                            continue
                        candidates_set.append(xi)
                        r = random.random()
                        px = gtp[xi]
                        qx = cart_candidates_prob[j, i]
                        if qx <= 0:
                            continue
                        acp = px / qx
                        if r <= acp:
                            accept_cand = torch.cat((accept_cand, x[None]), dim=0)
                            accept_length += 1
                            best_candidate = j
                            break
                        
                        q = op[i - 1][p_indices[j][i]].clone()
                        b = b_indices[j][i]
                        if len(b) > 0:
                            mask = tree_candidates[0][b]
                            if mode == 'RRS_wo_replacement':
                                q[mask] = 0
                            q = q / q.sum()
                        gtp = gtp - q
                        gtp[gtp < 0] = 0
                        gtp = gtp / gtp.sum()
                        adjustflag = True
        if adjustflag and accept_length != candidates.shape[1]:
            sample_p = gtp
        else:
            gt_logits = logits[best_candidate, accept_length - 1]
            sample_p = torch.softmax(gt_logits, dim=0)
        return torch.tensor(best_candidate), accept_length - 1, sample_p


@torch.no_grad()
def update_inference_inputs(
        input_ids,
        candidates,
        best_candidate,
        accept_length,
        retrieve_indices,
        logits_processor,
        logits,
        tree_logits,
        new_token,
        past_key_values_data_list,
        current_length_data,
        model,
        hidden_state,
        hidden_state_new,
        sample_p,
        mode
):
    prev_input_len = input_ids.shape[1]
    # Map the best candidate indices to the original indices in the sequence
    select_indices = (
            retrieve_indices[best_candidate, : accept_length + 1] + prev_input_len
    )
    # Append the tokens from the best candidate to the input sequence
    input_ids = torch.cat(
        [input_ids, candidates[None, best_candidate, : accept_length + 1].to(input_ids.device)], dim=-1
    )
    # Update the past key values based on the selected tokens
    # Source tensor that contains relevant past information based on the selected candidate
    for past_key_values_data in past_key_values_data_list:
        tgt = past_key_values_data[..., select_indices.to(past_key_values_data.device), :]
        # Destination tensor where the relevant past information will be stored
        dst = past_key_values_data[..., prev_input_len: prev_input_len + tgt.shape[-2], :]
        # Copy relevant past information from the source to the destination
        dst.copy_(tgt, non_blocking=True)

    # Update the current length tensor (currently only support batch size is 1)
    current_length_data.fill_(prev_input_len + tgt.shape[-2])

    retrieve_hidden_state_new = hidden_state_new[:, retrieve_indices]
    accept_hidden_state_new = retrieve_hidden_state_new[:, best_candidate, : accept_length + 1]
    # token=model.base_model.lm_head(accept_hidden_state_new[:,-1]).argmax()
    # token=token[None,None]
    prob = sample_p
    if logits_processor is not None:
        token = torch.multinomial(prob, 1)
        token = token[None]
    else:
        token = torch.argmax(prob)
        token = token[None, None]
    # hidden_state = torch.cat((hidden_state, accept_hidden_state_new), dim=1)
    tree_logits = model.ea_layer.topK_genrate(accept_hidden_state_new,
                                              input_ids=torch.cat((input_ids, token.to(input_ids.device)), dim=1),
                                              head=model.base_model.lm_head, logits_processor=logits_processor, 
                                              mode=mode)

    new_token += accept_length + 1

    return input_ids, tree_logits, new_token, None, token


if __name__ == "__main__":
    logits = torch.randn(1, 5)
    tp = prepare_logits_processor(0.9, 0, 0.9, 0)
    l = tp(None, logits)
    if tp is None:
        print(tp)
