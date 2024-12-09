from typing import List, Optional, Tuple, Union

import torch
import numpy as np
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

class RectFlowInverseScheduler(RectFlowScheduler):
    """
    RectFlowInverseScheduler is the reverse scheduler of [`RectFlowScheduler`].
    """
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
        pred_prev_sample = sample - model_output * dt
        pred_original_sample = sample - model_output * (timestep * dt)

        if not return_dict:
            return (
                pred_prev_sample,
                pred_original_sample,
            )

        return DDPMSchedulerOutput(prev_sample=pred_prev_sample, pred_original_sample=pred_original_sample)
        
    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device] = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """

        if num_inference_steps > self.config.num_train_timesteps:
            raise ValueError(
                f"`num_inference_steps`: {num_inference_steps} cannot be larger than `self.config.train_timesteps`:"
                f" {self.config.num_train_timesteps} as the unet model trained with this scheduler can only handle"
                f" maximal {self.config.num_train_timesteps} timesteps."
            )

        self.num_inference_steps = num_inference_steps
        step_ratio = self.config.num_train_timesteps // self.num_inference_steps
        # creates integer timesteps by multiplying by ratio
        # casting to int to avoid issues when num_inference_step is power of 3
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round().copy().astype(np.int64)
        self.timesteps = torch.from_numpy(timesteps).to(device)
        self.timesteps += self.config.steps_offset
        
