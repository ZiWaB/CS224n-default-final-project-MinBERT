from typing import Callable, Iterable, Tuple
import math

import torch
from sympy.abc import epsilon
from torch.optim import Optimizer

class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                """ 
                p: torch.tensor, the model parameters.
                p.grad: torch.tensor, the backpropagated gradients of p.
                """
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary.
                state = self.state[p]


                # Access hyperparameters from the `group` dictionary.
                alpha = group["lr"]

                # Complete the implementation of AdamW here, reading and saving
                # your state in the `state` dictionary above.
                # The hyperparameters can be read from the `group` dictionary
                # (they are lr, betas, eps, weight_decay, as saved in the constructor).
                #
                # To complete this implementation:
                # 1. Update the first and second moments of the gradients.
                # 2. Apply bias correction
                #    (using the "efficient version" given in https://arxiv.org/abs/1412.6980;
                #     also given in the pseudo-code in the project description).
                # 3. Update parameters (p.data).
                # 4. Apply weight decay after the main gradient-based updates.
                # Refer to the default project handout for more details.

                ### TODO-Done

                # Get the Hy-paras
                beta1, beta2 = group["betas"]
                epsilon = group["eps"]
                weight_decay = group["weight_decay"]

                # Initialize the states
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state["step"] = torch.tensor(0.0, dtype=torch.float64)
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # Get the stata at step: t - 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # Current step: t
                step_t = state["step"]
                step_t += 1

                # Compute m: exp_avg, and v: exp_avg_sq
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Compute bias corrections to the step size, denominator and numerator

                bias_correction_nu = (1 - beta2**step_t)**0.5
                bias_correction_de = 1 - beta1**step_t

                # Bias corrected step size
                alpha_bias_corrected = alpha * bias_correction_nu / bias_correction_de

                # Updates parameters
                p.data -= alpha_bias_corrected * exp_avg / (exp_avg_sq.sqrt() + epsilon) + alpha * weight_decay * p.data

                # print('state["exp_avg"] is', state["exp_avg"])
                # print('Current exp_avg is', exp_avg)


        return loss

