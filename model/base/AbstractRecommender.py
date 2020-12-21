import os
import time
from reckit import Logger
from data.dataset import Dataset
from utils import csr_to_user_dict
from reckit import Evaluator
import tensorflow as tf


class AbstractRecommender(object):
    def __init__(self, config):
        self.dataset = Dataset(config)
        self.logger = self._create_logger(config, self.dataset.name)
        self.logger.info("\nuser number=%d\nitem number=%d" % (self.dataset.num_users, self.dataset.num_items))

        user_train_dict = csr_to_user_dict(self.dataset.train_matrix)
        user_test_dict = csr_to_user_dict(self.dataset.test_matrix)

        self.evaluator = Evaluator(user_train_dict, user_test_dict,
                                   metric=config.metric, top_k=config.top_k,
                                   batch_size=config.test_batch_size,
                                   num_thread=config.test_thread)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.gpu_options.per_process_gpu_memory_fraction = config["gpu_mem"]
        self.sess = tf.Session(config=tf_config)

    def _create_logger(self, config, data_name):
        timestamp = time.time()
        if "pytorch" in self.__class__.__module__:
            model_name = "torch_" + self.__class__.__name__
        elif "tensorflow" in self.__class__.__module__:
            model_name = "tf_" + self.__class__.__name__
        else:
            model_name = self.__class__.__name__
        param_str = f"{data_name}_{model_name}_{config.summarize()}"
        run_id = f"{param_str[:150]}_{timestamp:.8f}"

        log_dir = os.path.join("log", data_name, self.__class__.__name__)
        logger_name = os.path.join(log_dir, run_id + ".log")
        logger = Logger(logger_name)

        logger.info(f"my pid: {os.getpid()}")
        logger.info(f"model: {self.__class__.__module__}")
        logger.info(self.dataset)
        logger.info(config)

        return logger

    def train_model(self):
        raise NotImplementedError
    
    def evaluate_model(self):
        raise NotImplementedError

    def predict(self, users):
        raise NotImplementedError
