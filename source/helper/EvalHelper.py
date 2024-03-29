import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from ranx import Qrels
from ranx import Run
from ranx import evaluate
from tqdm import tqdm
from tqdm.contrib import tzip

from source.helper.Helper import Helper


class EvalHelper:

    def __init__(self, params):
        self.params = params
        self.helper = Helper(params)
        self.relevance_map = self._load_relevance_map()
        self.labels_cls = self._load_labels_cls()
        self.texts_cls = self._load_texts_cls()
        self.metrics = self._get_metrics()

    def _get_metrics(self):
        metrics = []
        for metric in self.params.eval.metrics:
            for threshold in self.params.eval.thresholds:
                metrics.append(f"{metric}@{threshold}")
            metrics.append(f"{metric}@{self.params.data.num_relevant_labels}")

        return metrics

    # def _retrieve(self, prediction, ids_map, cls):
    #     ranking = {}
    #     rows, cols = prediction.nonzero()
    #     for row, col in tzip(rows, cols, desc="Ranking"):
    #         text_idx = ids_map[row]
    #         label_idx = col
    #         if (cls in self.labels_cls[label_idx] and cls in self.texts_cls[text_idx]) or cls == "all":
    #             if f"text_{text_idx}" in ranking:
    #                 ranking[f"text_{text_idx}"][f"label_{label_idx}"] = prediction[row, label_idx]
    #             else:
    #                 ranking[f"text_{text_idx}"] = {}
    #                 ranking[f"text_{text_idx}"][f"label_{label_idx}"] = prediction[row, label_idx]
    #     return ranking

    # def _retrieve(self, prediction, ids_map, cls):
    #     ranking = {}
    #     rows, cols = prediction.nonzero()
    #     for row in tqdm(set(rows), desc="Ranking"):
    #         text_idx = ids_map[row]
    #         if cls in self.texts_cls[text_idx]:
    #             labels_scores = {}
    #             for label_idx in set(cols):
    #                 if cls in self.labels_cls[label_idx]:
    #                     labels_scores[f"label_{label_idx}"] = prediction[row, label_idx]
    #             ranking[f"text_{text_idx}"] = labels_scores if len(labels_scores) > 0 else {"label_-1": 0.0}
    #
    #     return ranking

    def _retrieve(self, prediction, ids_map, cls):
        ranking = {}
        rows, cols = prediction.nonzero()
        for row, label_idx in tzip(rows, cols, desc=f"Ranking {cls} labels"):
        #for row, label_idx in tqdm(zip(rows, cols), desc="Ranking"):
            text_idx = ids_map[row]
            if cls in self.texts_cls[text_idx]:
                if f"text_{text_idx}" not in ranking:
                    ranking[f"text_{text_idx}"] = {"label_-1": 0.0}
                if cls in self.labels_cls[label_idx]:
                    ranking[f"text_{text_idx}"][f"label_{label_idx}"] = prediction[row, label_idx]

        # error = 0
        # for key, value in ranking.items():
        #     if len(value) < 1:
        #         error += 1
        #         print(f"\n\nERROR {key}\n\n")
        #
        # print(f"\n\nERROR {error}\n\n")

        return ranking

    def perform_eval(self):
        results = []
        rankings = {}

        for fold_idx in self.helper.params.data.folds:
            rankings[fold_idx] = {}
            print(
                f"Evaluating {self.params.model.name} over {self.params.data.name} (fold {fold_idx}) with fowling params\n"
                f"{OmegaConf.to_yaml(self.params)}\n")

            prediction = self.helper.load_prediction(fold_idx)
            ids_map = self._load_ids_map(fold_idx)

            for cls in self.params.eval.label_cls:
                ranking = self._retrieve(prediction, ids_map, cls)
                # print(f"Ranking ({len(ranking)}) for {cls}")
                filtered_dictionary = {key: value for key, value in self.relevance_map.items() if key in ranking.keys()}
                qrels = Qrels(filtered_dictionary)
                run = Run(ranking)
                result = evaluate(qrels, run, self.metrics)
                result = {k: round(v, 3) for k, v in result.items()}
                result["fold"] = fold_idx
                result["cls"] = cls

                rankings[fold_idx][cls] = ranking
                results.append(result)
            self.checkpoint_ranking(rankings[fold_idx], fold_idx)

        self.helper.checkpoint_results(results)

    def checkpoint_ranking(self, ranking, fold_idx):
        ranking_dir = f"{self.params.ranking.dir}{self.params.model.name}_{self.params.data.name}/"
        Path(ranking_dir).mkdir(parents=True, exist_ok=True)
        print(f"Saving ranking {fold_idx} on {ranking_dir}")
        with open(f"{ranking_dir}{self.params.model.name}_{self.params.data.name}_{fold_idx}.rnk",
                  "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def _load_relevance_map(self):
        with open(f"{self.params.data.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"label_{label_idx}"] = 1.0
            relevance_map[f"text_{text_idx}"] = d
        return relevance_map

    def _load_labels_cls(self):
        with open(f"{self.params.data.dir}label_cls.pkl", "rb") as label_cls_file:
            return pickle.load(label_cls_file)

    def _load_texts_cls(self):
        with open(f"{self.params.data.dir}text_cls.pkl", "rb") as text_cls_file:
            return pickle.load(text_cls_file)

    def _load_ids_map(self, fold_id):
        test_samples_df = self.helper.get_samples(fold_id=fold_id, split="test")
        return pd.Series(
            test_samples_df["text_idx"].values,
            index=test_samples_df.index
        ).to_dict()
