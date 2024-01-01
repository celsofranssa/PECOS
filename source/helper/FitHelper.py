from omegaconf import OmegaConf
from pecos.xmc.xtransformer.model import XTransformer
from pecos.xmc.xlinear.model import XLinearModel

from pecos.xmc.xtransformer.module import MLProblemWithText

from source.helper.Helper import Helper


class FitHelper:

    def __init__(self, params):
        self.helper = Helper(params)

    def perform_fit(self):
        for fold_id in self.helper.params.data.folds:
            print(
                f"Fitting {self.helper.params.model.name} over {self.helper.params.data.name} (fold {fold_id}) with fowling self.params\n"
                f"{OmegaConf.to_yaml(self.helper.params)}\n")

            print("Loading text and labels ...")
            texts, texts_rpr, labels_rpr = self.helper.get_texts_labels(fold_id, split="train")


            if self.helper.params.model.name == "XLinear":
                train_params = XLinearModel.TrainParams.from_dict(
                    OmegaConf.to_container(self.helper.params.model.train_params, resolve=True)
                )
                print("Training ... ")
                model = XLinearModel.train(
                    X=texts_rpr, Y=labels_rpr, train_params=train_params)
                self.helper.checkpoint_model(model, fold_id)

            elif self.helper.params.model.name == "XR-TFMR":
                problem = MLProblemWithText(X_text=texts, Y=labels_rpr, X_feat=texts_rpr)
                train_params = XTransformer.TrainParams.from_dict(
                    OmegaConf.to_container(self.helper.params.model.train_params, resolve=True)
                )
                model = XTransformer.train(problem, train_params=train_params)
                self.helper.checkpoint_model(model, fold_id)

            elif self.helper.params.model.name == "ANN":
                pass

