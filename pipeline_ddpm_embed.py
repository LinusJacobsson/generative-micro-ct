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
        z_embedded: Optional[torch.Tensor] = None,
        z_indices: Optional[torch.Tensor] = None,  # New parameter for z-coordinate indices


    ) -> Union[ImagePipelineOutput, Tuple]:
        
        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            # randn does not work reproducibly on mps
            image = randn_tensor(image_shape, generator=generator)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)

        # In the __call__ method of EmbeddedDDPMPipeline
        for t in self.progress_bar(self.scheduler.timesteps):
            if z_embedded is not None and z_indices is not None:
                z_expanded = z_embedded.expand(batch_size, -1, image.shape[2], image.shape[3])
                image_with_z = torch.cat([image, z_expanded], dim=1)
            else:
                image_with_z = image

            # Correctly pass the timesteps and z_indices arguments to the unet model
            model_output = self.unet(image_with_z, z_indices=z_indices, timesteps=t).sample

            print("Model output shape:", model_output.shape)  # Debugging print

            # 2. compute previous image: x_t -> x_t-1
            image = self.scheduler.step(model_output, t, image_with_z, generator=generator).prev_sample
            print("Updated image shape:", image.shape)  # Debugging print


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)