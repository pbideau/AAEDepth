from Estimators.PPPVelocityEstimator import PPPVelocityEstimator
from Estimators.LinearVelocityEstimator import LinearVelocityEstimator
from utils.utils import *

        
class PPPLinearVelocityEstimator(PPPVelocityEstimator, LinearVelocityEstimator):
    def __init__(self, fx, fy, px, py, dataset_path, sequence, Ne, height, width, dt=0.025, overlap=0, fixed_size = False, padding = 0,
                    optimizer = 'Adam', optim_kwargs = None, lr = 0.01, lr_step = 80, lr_decay = 0.1, iters = 80, sigma_prior=1, version=0, gamma_param = [0.1, 1.59]
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
        self.trans_type = "trans"



