# coding: utf-8
# @email: enoche.chow@gmail.com
"""
################################
"""
import os
import numpy as np
import pandas as pd
import torch
from utils.metrics import metrics_dict, item_metrics_dict, num_items_only_item_metrics
from torch.nn.utils.rnn import pad_sequence
from utils.utils import get_local_time


# These metrics are typical in topk recommendations, evaluated against user relevance
topk_metrics = {metric.lower(): metric for metric in ['Recall', 'Recall2', 'Precision', 'NDCG', 'MAP']}
# These "beyond-accuracy" metrics evaluate the distribution of recommended items
# across users (using item popularity computed from the training set) rather
# than their relevance to any single user
item_metrics = {metric.lower(): metric for metric in
                 ['GiniIndex', 'AveragePopularity', 'TailPercentage', 'ItemCoverage', 'ShannonEntropy']}
all_metrics = {**topk_metrics, **item_metrics}


class TopKEvaluator(object):
    r"""TopK Evaluator is mainly used in ranking tasks. Now, we support six topk metrics which
    contain `'Hit', 'Recall', 'MRR', 'Precision', 'NDCG', 'MAP'`.

    Note:
        The metrics used calculate group-based metrics which considers the metrics scores averaged
        across users. Some of them are also limited to k.

    """

    def __init__(self, config):
        self.config = config
        self.metrics = config['metrics']
        self.topk = config['topk']
        self.save_recom_result = config['save_recommended_topk']
        self.tail_ratio = config['tail_ratio'] if config['tail_ratio'] else 0.1
        self._check_args()
        # cache of (item_count, num_items) per training dataset, keyed by id() since
        # the training dataset is fixed for the lifetime of a trainer/evaluator
        self._item_popularity_cache = {}

    def collect(self, interaction, scores_tensor, full=False):
        """collect the topk intermediate result of one batch, this function mainly
        implements padding and TopK finding. It is called at the end of each batch

        Args:
            interaction (Interaction): :class:`AbstractEvaluator` of the batch
            scores_tensor (tensor): the tensor of model output with size of `(N, )`
            full (bool, optional): whether it is full sort. Default: False.

        """
        user_len_list = interaction.user_len_list
        if full is True:
            scores_matrix = scores_tensor.view(len(user_len_list), -1)
        else:
            scores_list = torch.split(scores_tensor, user_len_list, dim=0)
            scores_matrix = pad_sequence(scores_list, batch_first=True, padding_value=-np.inf)  # nusers x items

        # get topk
        _, topk_index = torch.topk(scores_matrix, max(self.topk), dim=-1)  # nusers x k

        return topk_index

    def evaluate(self, batch_matrix_list, eval_data, is_test=False, idx=0):
        """calculate the metrics of all batches. It is called at the end of each epoch

        Args:
            batch_matrix_list (list): the results of all batches
            eval_data (Dataset): the class of test data
            is_test: in testing?

        Returns:
            dict: such as ``{'Hit@20': 0.3824, 'Recall@20': 0.0527, 'Hit@10': 0.3153, 'Recall@10': 0.0329}``

        """
        pos_items = eval_data.get_eval_items()
        pos_len_list = eval_data.get_eval_len_list()
        topk_index = torch.cat(batch_matrix_list, dim=0).cpu().numpy()
        # if save recommendation result?
        if self.save_recom_result and is_test:
            dataset_name = self.config['dataset']
            model_name = self.config['model']
            max_k = max(self.topk)
            dir_name = os.path.abspath(self.config['recommend_topk'])
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
            file_path = os.path.join(dir_name, '{}-{}-idx{}-top{}-{}.csv'.format(
                model_name, dataset_name, idx, max_k, get_local_time()))
            x_df = pd.DataFrame(topk_index)
            x_df.insert(0, 'id', eval_data.get_eval_users())
            x_df.columns = ['id']+['top_'+str(i) for i in range(max_k)]
            x_df = x_df.astype(int)
            x_df.to_csv(file_path, sep='\t', index=False)
        assert len(pos_len_list) == len(topk_index)

        # compute each metric's result array (shape (max_k,)) keyed by metric
        # name first, then assemble metric_dict below following the order
        # metrics were declared in the config, regardless of which of the two
        # groups (relevance-based vs. item-distribution) they belong to
        results = {}

        # relevance-based (accuracy) metrics
        if self.topk_metrics:
            # if recom right?
            bool_rec_matrix = []
            for m, n in zip(pos_items, topk_index):
                bool_rec_matrix.append([True if i in m else False for i in n])
            bool_rec_matrix = np.asarray(bool_rec_matrix)

            result_list = self._calculate_metrics(pos_len_list, bool_rec_matrix)
            results.update(zip(self.topk_metrics, result_list))

        # beyond-accuracy (item-distribution) metrics
        if self.item_metrics:
            item_count, num_items = self._get_item_popularity(eval_data)
            result_list = self._calculate_item_metrics(topk_index, item_count, num_items)
            results.update(zip(self.item_metrics, result_list))

        metric_dict = {}
        for metric in self.metrics:
            value = results[metric]
            for k in self.topk:
                key = '{}@{}'.format(metric, k)
                metric_dict[key] = round(value[k - 1], 4)

        return metric_dict

    def _get_item_popularity(self, eval_data):
        """Item popularity (interaction count) computed from the training set,
        as a dense (num_items,) array, plus the catalog size. Cached per
        training dataset since it does not change across evaluation calls.
        """
        train_dataset = eval_data.additional_dataset
        key = id(train_dataset)
        if key not in self._item_popularity_cache:
            iid_field = train_dataset.iid_field
            num_items = train_dataset.item_num
            counts = train_dataset.df[iid_field].value_counts()
            item_count = np.zeros(num_items, dtype=np.int64)
            item_count[counts.index.values.astype(int)] = counts.values
            self._item_popularity_cache[key] = (item_count, num_items)
        return self._item_popularity_cache[key]

    def _check_args(self):
        # Check metrics
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                self.metrics = [self.metrics]
        else:
            raise TypeError('metrics must be str or list')

        # Convert metric to lowercase
        for m in self.metrics:
            if m.lower() not in all_metrics:
                raise ValueError("There is no user grouped topk metric named {}!".format(m))
        self.metrics = [metric.lower() for metric in self.metrics]
        # split into relevance-based vs. item-distribution (beyond-accuracy) metrics,
        # since they need different inputs to compute
        self.topk_metrics = [m for m in self.metrics if m in topk_metrics]
        self.item_metrics = [m for m in self.metrics if m in item_metrics]

        # Check topk:
        if isinstance(self.topk, (int, list)):
            if isinstance(self.topk, int):
                self.topk = [self.topk]
            for topk in self.topk:
                if topk <= 0:
                    raise ValueError(
                        'topk must be a positive integer or a list of positive integers, but get `{}`'.format(topk))
        else:
            raise TypeError('The topk must be a integer, list')

    def _calculate_metrics(self, pos_len_list, topk_index):
        """integrate the results of each batch and evaluate the topk metrics by users

        Args:
            pos_len_list (list): a list of users' positive items
            topk_index (np.ndarray): a matrix which contains the index of the topk items for users
        Returns:
            np.ndarray: a matrix which contains the metrics result
        """
        result_list = []
        for metric in self.topk_metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_index, pos_len_list)
            result_list.append(result)
        return np.stack(result_list, axis=0)

    def _calculate_item_metrics(self, topk_index, item_count, num_items):
        """calculate the beyond-accuracy (item-distribution) metrics

        Args:
            topk_index (np.ndarray): matrix of recommended item ids, (n_users, max_k)
            item_count (np.ndarray): training-set interaction count per item, (num_items,)
            num_items (int): total number of items in the catalog
        Returns:
            list: one np.ndarray per metric, each of shape (max_k,)
        """
        result_list = []
        for metric in self.item_metrics:
            metric_fuc = item_metrics_dict[metric]
            if metric in num_items_only_item_metrics:
                result = metric_fuc(topk_index, num_items)
            elif metric == 'tailpercentage':
                result = metric_fuc(topk_index, item_count, num_items, self.tail_ratio)
            else:
                result = metric_fuc(topk_index, item_count, num_items)
            result_list.append(result)
        return result_list

    def __str__(self):
        mesg = 'The TopK Evaluator Info:\n' + '\tMetrics:[' + ', '.join(
            [all_metrics[metric.lower()] for metric in self.metrics]) \
               + '], TopK:[' + ', '.join(map(str, self.topk)) + ']'
        return mesg
