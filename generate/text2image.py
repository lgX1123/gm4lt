from PIL import Image
import torch as th
import sys
import numpy as np

from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler
)

# multi GPU implementation of generating synthetic images
num_gpu = 2

def main(id, text_path, save_path):

    has_cuda = th.cuda.is_available()
    device = th.device('cpu' if not has_cuda else 'cuda')

    # Create base model.
    options = model_and_diffusion_defaults()
    options['use_fp16'] = has_cuda
    options['timestep_respacing'] = '100' # use 100 diffusion steps for fast sampling
    model, diffusion = create_model_and_diffusion(**options)
    model.eval()
    if has_cuda:
        model.convert_to_fp16()
    model.to(device)
    model.load_state_dict(load_checkpoint('base', device))
    print('total base parameters', sum(x.numel() for x in model.parameters()))

    def transform(image: th.Tensor):
        """ Transform image to (32, 32, 3) """
        scaled = ((image + 1)*127.5).round().clamp(0,255).to(th.uint8).cpu()
        original_image_np = scaled.permute(2, 0, 3, 1).reshape([image.shape[2], -1, 3]).numpy()
        original_image_pil = Image.fromarray(np.uint8(original_image_np))
        resized_image = original_image_pil.resize((32, 32), Image.LANCZOS)
        resized_image_np = np.array(resized_image)
        return resized_image_np

    with open(text_path, 'r') as f:
        prompt_list = f.readlines()

    batch_size = 1
    guidance_scale = 3.0
    data = np.zeros((50000 // num_gpu, 32, 32, 3)).astype(np.uint8)

    total_len = len(prompt_list)

    if total_len % num_gpu == 0:
        each_len = total_len // num_gpu
    else:
        each_len = total_len // num_gpu +1

    if id != num_gpu-1:
        prompt_list = prompt_list[id*each_len:(id+1)*each_len]
        print('{}:{}'.format(id*each_len,(id+1)*each_len))
    else:
        prompt_list = prompt_list[id * each_len:]
        print('{}:'.format(id * each_len))

    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = th.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = th.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = th.cat([half_eps, half_eps], dim=0)
        return th.cat([eps, rest], dim=1)
    
    for index, prompt in enumerate(prompt_list):
        ##############################
        # Sample from the base model #
        ##############################

        # Create the text tokens to feed to the model.
        tokens = model.tokenizer.encode(prompt)
        tokens, mask = model.tokenizer.padded_tokens_and_mask(
            tokens, options['text_ctx']
        )

        # Create the classifier-free guidance tokens (empty)
        full_batch_size = batch_size * 2
        uncond_tokens, uncond_mask = model.tokenizer.padded_tokens_and_mask(
            [], options['text_ctx']
        )

        # Pack the tokens together into model kwargs.
        model_kwargs = dict(
            tokens=th.tensor(
                [tokens] * batch_size + [uncond_tokens] * batch_size, device=device
            ),
            mask=th.tensor(
                [mask] * batch_size + [uncond_mask] * batch_size,
                dtype=th.bool,
                device=device,
            ),
        )

        # Sample from the base model.
        model.del_cache()
        samples = diffusion.p_sample_loop(
            model_fn,
            (full_batch_size, 3, options["image_size"], options["image_size"]),
            device=device,
            clip_denoised=True,
            progress=True,
            model_kwargs=model_kwargs,
            cond_fn=None,
        )[:batch_size]
        data[index] = transform(samples)
        model.del_cache()

    np.save(save_path + '{}.npy'.format(id), data)


if __name__ == "__main__":
    import os
    import sys
    id = sys.argv[1]
    text_path = sys.argv[2]
    save_path = sys.argv[3]
    print(sys.argv[1])
    id = int(id)

    main(id, text_path, save_path)






