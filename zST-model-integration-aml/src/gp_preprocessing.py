import os
import json
import logging
import time
import numpy as np
import logging


class GpPreprocessing:
    def __init__(self, transformed_data_path: str):
        self.transformed_data: np.array = np.load(transformed_data_path)
        if (self.transformed_data is None or len(self.transformed_data) <= 0):
            raise Exception("Transformed data is missing. Please configure")
        logging.info("GP pre-processed transformed data loaded")

    def preprocess(self, indexes: [int]):
        tt0 = time.time()
        features_out = np.take(self.transformed_data, indexes, 0)
        tt1 = time.time()
        elapsed_time = tt1-tt0
        return (features_out, elapsed_time)
