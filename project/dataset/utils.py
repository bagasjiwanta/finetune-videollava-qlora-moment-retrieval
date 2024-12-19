from matplotlib import pyplot as plt
from matplotlib import animation
from IPython.display import HTML
from PIL import Image
import numpy as np

def display_videos(pixels):
    # convert to image from proceessed tensors
    clip = pixels[0] * 255
    clip = clip.permute(0, 2, 3, 1).clamp(0, 255)

    # np array with shape (frames, height, width, channels)
    video = np.array(clip).astype(np.uint8)

    fig = plt.figure()
    im = plt.imshow(video[0,:,:,:])

    plt.close() # this is required to not display the generated image

    def init():
        im.set_data(video[0,:,:,:])

    def animate(i):
        im.set_data(video[i,:,:,:])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video.shape[0],
                                interval=100)
    display(HTML(anim.to_html5_video()))


def display_images_from_video(pixels, frames):
    clip = pixels[0] * 255
    clip = clip.permute(0, 2, 3, 1).clamp(0, 255)
    fig, axarr = plt.subplots(2, frames//2, figsize = (12, 12))
    fig.tight_layout()

    for i in range(2):
        for j in range(frames // 2):
            curr_frame = Image.fromarray(np.uint8(clip[i + j]))
            axarr[i, j].imshow(curr_frame)
            axarr[i, j].get_xaxis().set_visible(False)
            axarr[i, j].get_yaxis().set_visible(False)
            axarr[i, j].set_aspect('equal')

    plt.subplots_adjust(wspace=None, hspace=None)
    plt.axis('off')
    plt.show()


def view_sample_with_video(sample, processor):
    display_videos(sample['pixel_values_videos'])
    print('prompt:')
    print(processor.batch_decode(sample['input_ids'], clean_up_tokenization_space=True, skip_special_tokens=True)[0])

    if 'answer' in sample:
        print(f"answer:\n{sample['answer']}")
    if 'ts_info' in sample:
        print(f"ts_info: \n{sample['ts_info']}")


def view_sample_with_video_images(sample, processor, num_frames):
    display_images_from_video(sample['pixel_values_videos'], num_frames)
    print('prompt:')
    print(processor.batch_decode(sample['input_ids'], clean_up_tokenization_space=True, skip_special_tokens=True)[0])

    if 'answer' in sample:
        print(f"answer:\n{sample['answer']}")
    if 'ts_info' in sample:
        print(f"ts_info: \n{sample['ts_info']}")