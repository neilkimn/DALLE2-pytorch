from dalle2_pytorch import DALLE2, OpenAIClipAdapter, OpenClipAdapter

import clip_training
import decoder_training
import diffusion_prior_network
import time

if __name__ == "__main__":
    # clip = clip_training.get_trained_clip()
    # we can use the already trained "clip" from OpenAIClipAdapter or OpenClipAdapter
    clip = OpenAIClipAdapter()
    #clip = OpenClipAdapter('ViT-H/14', pretrained='laion2b_s32b_b79k')

    text, images = decoder_training.get_data()
    images, _ = clip.embed_image(images)
    text, _ = clip.embed_text(text)
    while True:
        print("zzz")
        time.sleep(1)

    decoder = decoder_training.get_trained_decoder(clip)
    diffusion_prior = diffusion_prior_network.get_prior_network(clip)
    dalle2 = DALLE2(
        prior=diffusion_prior,
        decoder=decoder,
    )

    # send the text as a string if you want to use the simple tokenizer from DALLE v1
    # or you can do it as token ids, if you have your own tokenizer

    texts = ['glistening morning dew on a flower petal']
    # texts = [input()]
    # images = dalle2(texts)  # (1, 3, 256, 256)
    images = dalle2(
        texts,
        cond_scale=2.,  # classifier free guidance strength (> 1 would strengthen the condition)
        return_pil_images=True,
    )
    print(images)
    images[0].save("test.jpg")