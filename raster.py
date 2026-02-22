import random
import argparse
import importlib
from itertools import product
from functools import partial
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.base import Color

class Context:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def render_pixel(context, ij):
    i, j = ij
    pixel = Color(0, 0, 0)
    for _ in range(context.num_samples):
        # random offset for anti-aliasing
        dx = np.random.uniform(-0.5, 0.5)
        dy = np.random.uniform(-0.5, 0.5)
        # middle of pixel coordinates
        x = j + 0.5 + dx
        y = i + 0.5 + dy
        # ray from camera
        ray = context.camera.ray(x, y)
        # hit ray with scene
        hit_rec = context.scene.hit(ray)
        # test if hit something
        if hit_rec.hit:
            # Simple shading: use the red channel as intensity
            material = hit_rec.material
            shaded_color = material.shade(hit_rec, context.scene)
            # this is box filtering!
            pixel = pixel + shaded_color / context.num_samples
        else:
            # this is box filtering!
            pixel = pixel + context.scene.background / context.num_samples
    return (i, j, pixel)

def main(args, pool):
    # load scene from file args.scene
    scene = importlib.import_module(args.scene).Scene()
    camera = scene.camera
    img_width = camera.img_width
    img_height = camera.img_height
    image = np.zeros((img_height, img_width, 3)) # create tensor for image: RGB

    # for each pixel, determine if it is inside any primitive in the scene
    # use cartesian product for efficiency
    print("Rendering... with anti-aliasing samples:", args.num_samples)
    context = Context(scene=scene, camera=camera, num_samples=args.num_samples)
    with tqdm(total=img_height*img_width) as pbar:
        if args.num_jobs <= 1:
            for i, j in product(range(img_height), range(img_width)):
                _, _, pixel = render_pixel(context, (i, j))
                image[i, j] = np.clip(pixel.as_list(), 0, 1)
                pbar.update(1)
                pbar.refresh()
        else:
            for i, j, pixel in pool.imap(partial(render_pixel, context), product(range(img_height), range(img_width))):
                image[i, j] = np.clip(pixel.as_list(), 0, 1)
                pbar.update(1)
                pbar.refresh()

    # save image as png using matplotlib
    plt.imsave(args.output, image, vmin=0, vmax=1, origin='lower')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Raster module main function")
    parser.add_argument('-s', '--scene', type=str, help='Scene name', default='ball_scene')
    parser.add_argument('-n', '--num_samples', type=int, help='Number of samples per pixel for anti-aliasing', default=1)
    parser.add_argument('-j', '--num_jobs', type=int, help='Number of parallel jobs for rendering', default=4)
    parser.add_argument('-o', '--output', type=str, help='Output image file name', default='output.png')
    args = parser.parse_args()

    # create a pool of workers for parallel processing
    pool = Pool(args.num_jobs)
    main(args, pool)
    pool.close()
    pool.join()