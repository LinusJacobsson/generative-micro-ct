from typing import List, Optional, Tuple, Union
import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

class EmbeddedDDPMPipeline(DiffusionPipeline):
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        z_indices: Optional[torch.Tensor] = None,
    ) -> Union[ImagePipelineOutput, Tuple]:
        # Generate the initial noise tensor
        image_shape = (
            batch_size,
            1,  # Ensure only one channel
            self.unet.config.sample_size,
            self.unet.config.sample_size,
        )
        image = randn_tensor(image_shape, generator=generator, device=self.device)

        # Embed the z_indices if provided
        if z_indices is not None:
            z_embedded = self.unet.z_embedding(z_indices)
            z_embedded = z_embedded.view(z_embedded.shape[0], z_embedded.shape[1], 1, 1)
            z_embedded = z_embedded.expand(-1, -1, image.shape[2], image.shape[3])
            image_with_z = torch.cat([image, z_embedded], dim=1)
        else:
            image_with_z = image

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            # Correctly pass the timesteps and z_indices arguments to the unet model
            model_output = self.unet(image_with_z, z_indices=z_indices, timesteps=t).sample

            # compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image_with_z, generator=generator).prev_sample

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
