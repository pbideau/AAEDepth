from Estimators.PPPAngularVelocityEstimator import PPPAngularVelocityEstimator
from Estimators.PPPLinearVelocityEstimator import PPPLinearVelocityEstimator
from Estimators.CMaxAngularVelocityEstimator import CMaxAngularVelocityEstimator
from Estimators.CMaxLinearVelocityEstimator import CMaxLinearVelocityEstimator

class EstimatorFactory(object):
    def __init__(self, fx, fy, px, py, method, transformation, dataset_path, sequence, Ne, height, width, dt=0.025, overlap=0, padding = 100,
                    lr = 0.05, iters = 250, sigma_prior=1, version=0, gamma_param = [0.1, 1.59]
                    ) -> None:
        if method == "st-ppp":
            if transformation == "rot":
                print("Using method: Poisson Point Process for angular velocity estimation")
                self.VE = PPPAngularVelocityEstimator(fx=fx, fy=fy, px=px, py=py, dataset_path=dataset_path, sequence=sequence, Ne=Ne, height=height, width=width, dt=dt, overlap=overlap, fixed_size=True, padding=padding,
                                                      lr=lr, lr_step=iters, iters=iters, sigma_prior=sigma_prior, version=version, gamma_param=gamma_param)
            elif transformation == "trans":
                print("Using method: Poisson Point Process for linear velocity estimation")
                self.VE = PPPLinearVelocityEstimator(fx=fx, fy=fy, px=px, py=py, dataset_path=dataset_path, sequence=sequence, Ne=Ne, height=height, width=width, dt=dt, overlap=overlap, fixed_size=True, padding=padding,
                                                     lr=lr, lr_step=iters, iters=iters, sigma_prior=sigma_prior, version=version, gamma_param=gamma_param)
            else:
                print("The transformation is not supported, please read help")
                exit()

        elif method == "cmax":
            if transformation == "rot":
                print("Using method: Contrast maximization for angular velocity estimation")
                self.VE = CMaxAngularVelocityEstimator(fx=fx, fy=fy, px=px, py=py, dataset_path=dataset_path, sequence=sequence, Ne=Ne, overlap=overlap, fixed_size=True, padding=padding, lr=lr, lr_step=iters, iters=iters)
            elif transformation == "trans":
                print("Using method: Contrast maximization for linear velocity estimation")
                self.VE = CMaxLinearVelocityEstimator(fx=fx, fy=fy, px=px, py=py, dataset_path=dataset_path, sequence=sequence, Ne=Ne, overlap=overlap, fixed_size=True, padding=padding, lr=lr, lr_step=iters, iters=iters)
            else:
                print("The transformation is not supported, please read help")
                exit()
        
        else:
            print("The method is not supported, please read help")
            exit()

    def __del__(self):
        print('Destructor called, Employee deleted.')

    def get_estimator(self):
        return self.VE
