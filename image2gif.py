import imageio
def img2gif(iterations):
    images = []
    filenames = ['gif/foo{}.png'.format(i) for i in range(iterations)]

    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('graphs/foo.gif',images)