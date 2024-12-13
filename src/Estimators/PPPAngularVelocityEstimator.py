from Estimators.PPPVelocityEstimator import PPPVelocityEstimator
from Estimators.AngularVelocityEstimator import AngularVelocityEstimator
from utils.utils import *


class PPPAngularVelocityEstimator(PPPVelocityEstimator, AngularVelocityEstimator):
    def __init__(self, fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt=0.025, overlap=0, fixed_size = True, padding = 100,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.05, lr_step = 250, lr_decay = 0.1, iters = 250, sigma_prior=1,
                    version=0, gamma_param = [0.1, 1.59]
                    ) -> None:
        super().__init__(   fx,
                            fy,
                            px,
                            py,
                            dataset_path, 
                            sequence, 
                            Ne,
                            height,
                            width,
                            dt,
                            overlap, 
                            fixed_size, 
                            padding, 
                            optimizer, 
                            optim_kwargs, 
                            lr, 
                            lr_step, 
                            lr_decay, 
                            iters,
                            sigma_prior,
                            version,
                            gamma_param)
        self.trans_type = "rot"

    def reset(self, fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt, overlap, padding, lr, iters,
              sigma_prior, version, gamma_param):
        self.__init__(fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt, overlap, True, padding,
                      'Adam', None, lr, 250, 0.1, iters, sigma_prior, version, gamma_param)