import pickle

import numpy as np
import pandas as pd
from pecos.utils import smat_util
from scipy import sparse
import scipy

#
# def load_prediction(model, data, fold_idx=0):
#     with open(f"resource/prediction/{model}_{data}_{fold_idx}.prd", "rb") as prediction_file:
#         return pickle.load(prediction_file)
#
# def load_samples(dataset):
#     with open(f"resource/dataset/{dataset}/samples.pkl", "rb") as samples_file:
#         return pickle.load(samples_file)
#
# def get_samples(dataset, fold_id, split):
#     ids = load_ids(dataset, fold_id, split)
#     samples_df = pd.DataFrame(load_samples(dataset))
#     return samples_df[samples_df["idx"].isin(ids)]
#
# def load_ids(dataset, fold_id, split):
#     with open(f"resource/dataset/{dataset}/fold_{fold_id}/{split}.pkl", "rb") as ids_file:
#         return pickle.load(ids_file)
#
# def load_vectorizer(dataset,fold_id):
#     with open(f"resource/dataset/{dataset}/fold_{fold_id}/vectorizer.pkl", "rb") as vectorizer_file:
#         return pickle.load(vectorizer_file)
#
# def get_texts_labels(dataset, fold_id, split, num_labels):
#     samples_df = get_samples(dataset=dataset, fold_id=fold_id, split=split)
#     vectorizer = load_vectorizer(dataset=dataset, fold_id=fold_id)
#
#     rows, cols, data = [], [], []
#
#     for row_idx, row in samples_df.iterrows():
#         for label_idx in row["label_ids"]:
#             rows.append(row_idx)
#             cols.append(label_idx)
#             data.append(1.0)
#
#     texts_rpr = sparse.csr_matrix(vectorizer.transform(samples_df["text"]), dtype=np.float32)
#     labels_rpr = sparse.csr_matrix((data, (rows, cols)), shape=(samples_df.shape[0], num_labels),
#                                    dtype=np.float32)
#     texts_rpr.sort_indices()
#     labels_rpr.sort_indices()
#     return samples_df["text"].tolist(), texts_rpr, labels_rpr
#
# fold_idx=0
# model="XR-TFMR"
# dataset="Eurlex57k"
#
# prediction = load_prediction(model, dataset.upper())
# texts, texts_rpr, labels_rpr = get_texts_labels(dataset=dataset, fold_id=0, split="test")
#
#
# metric = smat_util.Metrics.generate(labels_rpr, prediction, topk=10)
# print(metric)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n - 1)
    return f"{round(100 * m, 1)}({round(100 * h, 1)})"

def ic(model, dataset):
    print(f"{model}: {dataset}")
    result_df = pd.read_csv(f"resource/result/{model}_{dataset}.rts", header=0, sep="\t")
    for cls in ["tail", "head"]:
        print(f"Results for {cls}")
        cls_df = result_df[result_df["cls"] == cls]
        for metric in ['mrr@1', 'mrr@5', 'mrr@10', 'ndcg@1', 'ndcg@5', 'ndcg@10']:
            print(f"{metric}: {mean_confidence_interval(cls_df[metric])}")

def read_prediction(model, dataset, fold_idx):
    with open(f"resource/prediction/{model}_{dataset}_{fold_idx}.prd", "rb") as prediction_file:
        prediction = pickle.load(prediction_file)
    print()

if __name__ == '__main__':
    #read_prediction(model="XLinear", dataset="Wiki10-31k", fold_idx=0)
    ic(model="XR-TFMR", dataset="Wiki10-31k")
