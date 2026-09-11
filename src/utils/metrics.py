# encoding: utf-8
# @email: enoche.chow@gmail.com
"""
############################
"""

from logging import getLogger

import numpy as np


def recall_(pos_index, pos_len):
    # Recall: average single users recall ratio.
    rec_ret = np.cumsum(pos_index, axis=1) / pos_len.reshape(-1, 1)
    return rec_ret.mean(axis=0)


def recall2_(pos_index, pos_len):
    r"""
    All hits are summed up and then averaged for recall.
    :param pos_index:
    :param pos_len:
    :return:
    """
    rec_cum = np.cumsum(pos_index, axis=1)
    rec_ret = rec_cum.sum(axis=0) / pos_len.sum()
    return rec_ret


def ndcg_(pos_index, pos_len):
    r"""NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.
    .. _NDCG: https://en.wikipedia.org/wiki/Discounted_cumulative_gain#Normalized_DCG

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in u^{te}NDCG_u@K}}{|u^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`2^{rel_i}` equals to 1 if the item hits otherwise 0.
    :math:`U^{te}` is for all users in the test set.
    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float64)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float64)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result.mean(axis=0)


def map_(pos_index, pos_len):
    r"""MAP_ (also known as Mean Average Precision) The MAP is meant to calculate Avg. Precision for the relevant items.
    Note:
        In this case the normalization factor used is :math:`\frac{1}{\min (m,N)}`, which prevents your AP score from
        being unfairly suppressed when your number of recommendations couldn't possibly capture all the correct ones.

    .. _map: http://sdsawtelle.github.io/blog/output/mean-average-precision-MAP-for-recommender-systems.html#MAP-for-Recommender-Algorithms

    .. math::
        \begin{align*}
        \mathrm{AP@N} &= \frac{1}{\mathrm{min}(m,N)}\sum_{k=1}^N P(k) \cdot rel(k) \\
        \mathrm{MAP@N}& = \frac{1}{|U|}\sum_{u=1}^{|U|}(\mathrm{AP@N})_u
        \end{align*}
    """
    pre = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    sum_pre = np.cumsum(pre * pos_index.astype(np.float64), axis=1)
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    actual_len = np.where(pos_len > len_rank, len_rank, pos_len)
    result = np.zeros_like(pos_index, dtype=np.float64)
    for row, lens in enumerate(actual_len):
        ranges = np.arange(1, pos_index.shape[1]+1)
        ranges[lens:] = ranges[lens - 1]
        result[row] = sum_pre[row] / ranges
    return result.mean(axis=0)


def precision_(pos_index, pos_len):
    r"""Precision_ (also called positive predictive value) is the fraction of
    relevant instances among the retrieved instances
    .. _precision: https://en.wikipedia.org/wiki/Precision_and_recall#Precision

    .. math::
        \mathrm {Precision@K} = \frac{|Rel_u \cap Rec_u|}{Rec_u}

    :math:`Rel_u` is the set of items relavent to user :math:`U`,
    :math:`Rec_u` is the top K items recommended to users.
    We obtain the result by calculating the average :math:`Precision@K` of each user.
    """
    rec_ret = pos_index.cumsum(axis=1) / np.arange(1, pos_index.shape[1] + 1)
    return rec_ret.mean(axis=0)


"""Function name and function mapper.
Useful when we have to serialize evaluation metric names
and call the functions based on deserialized names
"""
metrics_dict = {
    'ndcg': ndcg_,
    'recall': recall_,
    'recall2': recall2_,
    'precision': precision_,
    'map': map_,
}


"""
############################
Beyond-accuracy (item-distribution) metrics.

Unlike the metrics above, these do not measure whether recommended items are
relevant to a user. They measure properties of the *distribution* of items
recommended across the whole user base, so they need the actual recommended
item ids (not just a relevance mask) plus item popularity computed from the
training interactions. Implementations follow RecBole's definitions
(recbole.evaluator.metrics) so that results are directly comparable.
"""


def giniindex_(item_matrix, num_items):
    r"""GiniIndex_ measures the inequality of the item recommendation frequency
    distribution: 0 means every item is recommended equally often, 1 means a
    single item absorbs every recommendation slot. Lower is more equitable.

    .. _GiniIndex: https://en.wikipedia.org/wiki/Gini_coefficient

    .. math::
        \mathrm {GiniIndex@K}=\frac{\sum_{i=1}^{|I|}(2i-|I|-1) P_{(i)}}{|I| \sum_{i=1}^{|I|} P_{(i)}}

    :math:`P_{(i)}` is the number of times item :math:`i` appears across all
    users' top-K lists, items indexed in non-decreasing order of :math:`P`.
    Items never recommended contribute 0 to the count but still occupy the
    lowest ranks of the distribution.

    :param item_matrix: (n_users, max_k) matrix of recommended item ids.
    :param num_items: total number of items in the catalog (|I|).
    """
    max_k = item_matrix.shape[1]
    result = np.zeros(max_k)
    cum_counts = np.zeros(num_items, dtype=np.int64)
    idx = np.arange(1, num_items + 1)
    for k in range(max_k):
        cum_counts += np.bincount(item_matrix[:, k], minlength=num_items)
        sorted_count = np.sort(cum_counts)
        total_num = sorted_count.sum()
        result[k] = np.sum((2 * idx - num_items - 1) * sorted_count) / (num_items * total_num)
    return result


def averagepopularity_(item_matrix, item_count, num_items):
    r"""AveragePopularity_ computes the average training-set popularity of the
    recommended items, averaged over users. Higher means the model leans on
    already-popular items.

    .. math::
        \mathrm {AveragePopularity@K}=\frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in R_u} \phi(i)}{|R_u|}

    :math:`\phi(i)` is the number of interactions of item :math:`i` in the
    training data (0 if the item never appeared in training).

    :param item_matrix: (n_users, max_k) matrix of recommended item ids.
    :param item_count: (num_items,) array, training-set interaction count per item.
    :param num_items: total number of items in the catalog (unused, kept for a
        uniform signature with the other item-distribution metrics).
    """
    pop = item_count[item_matrix].astype(np.float64)
    cum_avg = pop.cumsum(axis=1) / np.arange(1, item_matrix.shape[1] + 1)
    return cum_avg.mean(axis=0)


def tailpercentage_(item_matrix, item_count, num_items, tail_ratio=0.1):
    r"""TailPercentage_ (a.k.a. Average Percentage of Long-tail items, APLT)
    computes the fraction of recommended items that belong to the long tail,
    averaged over users. Higher means the model surfaces more long-tail items.

    .. _TailPercentage: https://en.wikipedia.org/wiki/Long_tail#Criticisms

    .. math::
        \mathrm {TailPercentage@K}=\frac{1}{|U|} \sum_{u \in U} \frac{\sum_{i \in R_u} \delta(i \in T)}{|R_u|}

    :math:`T` is the set of long-tail items: the least-popular items in the
    training data. If ``tail_ratio`` is in (0, 1], :math:`T` is the bottom
    ``tail_ratio`` fraction (by training popularity, ties broken by item id)
    of items that were seen at least once in training. If ``tail_ratio`` > 1,
    :math:`T` is every item whose training popularity is <= ``tail_ratio``.
    Items never seen in training are not counted as long-tail (they are a
    cold-start/coverage issue, not a popularity-bias one).

    :param item_matrix: (n_users, max_k) matrix of recommended item ids.
    :param item_count: (num_items,) array, training-set interaction count per item.
    :param num_items: total number of items in the catalog.
    :param tail_ratio: threshold defining the long tail, see above. Default 0.1.
    """
    observed = np.nonzero(item_count > 0)[0]
    if tail_ratio > 1:
        tail_mask = (item_count > 0) & (item_count <= tail_ratio)
    else:
        order = np.lexsort((observed, item_count[observed]))
        cut = max(int(len(observed) * tail_ratio), 1)
        tail_items = observed[order][:cut]
        tail_mask = np.zeros(num_items, dtype=bool)
        tail_mask[tail_items] = True
    is_tail = tail_mask[item_matrix].astype(np.float64)
    cum_avg = is_tail.cumsum(axis=1) / np.arange(1, item_matrix.shape[1] + 1)
    return cum_avg.mean(axis=0)


def itemcoverage_(item_matrix, num_items):
    r"""ItemCoverage_ measures the fraction of the item catalog that has been
    recommended to at least one user within the top-K. Higher means the
    model surfaces a larger share of the catalog.

    .. _ItemCoverage: https://en.wikipedia.org/wiki/Long_tail#Criticisms

    .. math::
        \mathrm {ItemCoverage@K}=\frac{|\bigcup_{u \in U} R_u@K|}{|I|}

    :math:`R_u@K` is the set of items recommended to user :math:`u` within
    the top-K, and :math:`|I|` is the catalog size.

    :param item_matrix: (n_users, max_k) matrix of recommended item ids.
    :param num_items: total number of items in the catalog (|I|).
    """
    max_k = item_matrix.shape[1]
    result = np.zeros(max_k)
    seen = np.zeros(num_items, dtype=bool)
    for k in range(max_k):
        seen[item_matrix[:, k]] = True
        result[k] = seen.sum() / num_items
    return result


def shannonentropy_(item_matrix, num_items):
    r"""ShannonEntropy_ measures the diversity of the item recommendation
    frequency distribution across users: 0 means every recommendation slot
    is filled by a single item, :math:`\ln(|I|)` is the maximum, reached when
    every item is recommended equally often. Higher is more diverse.

    .. _ShannonEntropy: https://en.wikipedia.org/wiki/Entropy_(information_theory)

    .. math::
        \mathrm {ShannonEntropy@K}=-\sum_{i=1}^{|I|} p_{(i)} \log p_{(i)}

    :math:`p_{(i)}=P_{(i)}/\sum_{j} P_{(j)}` is the fraction of all top-K
    recommendation slots occupied by item :math:`i`. Items never recommended
    (:math:`p_{(i)}=0`) contribute 0. Natural log is used, so results are in
    nats.

    :param item_matrix: (n_users, max_k) matrix of recommended item ids.
    :param num_items: total number of items in the catalog (|I|).
    """
    max_k = item_matrix.shape[1]
    result = np.zeros(max_k)
    cum_counts = np.zeros(num_items, dtype=np.int64)
    for k in range(max_k):
        cum_counts += np.bincount(item_matrix[:, k], minlength=num_items)
        probs = cum_counts[cum_counts > 0] / cum_counts.sum()
        result[k] = -np.sum(probs * np.log(probs))
    return result


"""Function name and function mapper for beyond-accuracy (item-distribution)
metrics. These take a different signature than `metrics_dict` above (they
need recommended item ids + item popularity, not a relevance mask), so they
are kept in a separate registry.
"""
item_metrics_dict = {
    'giniindex': giniindex_,
    'averagepopularity': averagepopularity_,
    'tailpercentage': tailpercentage_,
    'itemcoverage': itemcoverage_,
    'shannonentropy': shannonentropy_,
}

# item-distribution metrics whose implementation only needs the recommended
# item ids + catalog size (not per-item training popularity)
num_items_only_item_metrics = frozenset(('giniindex', 'itemcoverage', 'shannonentropy'))
