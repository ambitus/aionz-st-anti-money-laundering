import time
import numpy as np
from snapml import BoostingMachine
import lightgbm as lgb
import logging


class InMemoryModel:
    def __init__(self,
                 model_path="data/aml-e_100M_model.txt",
                 ZDNN=False,
                 use_snap_ml=False):
        logging.info("Loading BoostingMachine model...")
        self.use_snap_ml = use_snap_ml
        t0 = time.time()
        if (use_snap_ml):
            self.model = BoostingMachine()
            if ZDNN:
                self.model.import_model(model_path, "lightgbm", "zdnn_tensors")
            else:
                self.model.import_model(model_path, "lightgbm")
            self.model.set_param({"boosting_params": {"num_threads": 4}})
        else:
            self.model = lgb.Booster(model_file=model_path)
            self.model.params['objective'] = 'binary'
        t1 = time.time()
        logging.info(
            f"Successfully loaded model in {t1-t0} seconds")

    def get_prediction(self, rows: np.ndarray):
        try:
            tt0 = time.time()
            y_pred = self.model.predict(rows)
            tt1 = time.time()
            logging.info(f'Inference {rows.shape[0]} items in {tt1-tt0}')
            return y_pred
        except:
            logging.error('inference call error')
