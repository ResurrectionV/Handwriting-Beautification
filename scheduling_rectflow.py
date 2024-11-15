from typing import List, Optional, Tuple, Union

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput, DDPMScheduler


class RectFlowScheduler(DDPMScheduler):
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator=None,
        return_dict: bool = True,
    ) -> Union[DDPMSchedulerOutput, Tuple]:

        ### NOTE: Use Euler method to sample from the learned flow
        dt = 1. / self.num_inference_steps
        pred_prev_sample = sample + model_output * dt
        pred_original_sample = sample + model_output * (timestep * dt)

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:

        t = timesteps.view(-1, *([1] * (noise.ndim - 1))) / self.config.num_train_timesteps
        return t * noise + (1.-t) * original_samples

