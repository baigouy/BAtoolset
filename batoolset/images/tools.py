

def generate_random_number_images():
    # very good this generates random images for testing my tool or for building tutos maybe need some more love not to save the file but just get the raster and then maybe also force aspect ratio in some case or bg color for specific purposes --> TODO

    import numpy as np
    import matplotlib.pyplot as plt
    import random

    def generate_image_with_number(size, number, aspect_ratio=1.0):
        width = int(size * aspect_ratio)
        height = size

        # image = np.zeros((height, width, 3), dtype=np.uint8)

        # Set background color
        # image[:, :] = tuple([random.randint(0, 255) for _ in range(3)])

        # print(image)

        # Set the font properties
        font = {'family': 'sans-serif',
                'weight': 'bold',
                'size': 60}

        # Render the number on the image
        fig, ax = plt.subplots(figsize=(width / 100, height / 100))
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.text(0.5, 0.5, str(number),  fontdict=font,color='red', ha='center', va='center')
        plt.gca().set_aspect('equal', adjustable='datalim')

        # Render the figure and get the RGBA buffer
        fig.canvas.draw()
        # renderer = fig.canvas.get_renderer()
        rendered_image_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        rendered_image_rgba = rendered_image_rgba.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        # plt.show()

        # Close the figure to avoid memory leak
        plt.close(fig)

        # Convert RGBA to RGB format
        rendered_image = rendered_image_rgba[..., :3]

        # Replace the image with the rendered number

        image = np.zeros_like(rendered_image)
        image[:, :] = rendered_image


        # print(image)
        # image[image==[255,255, 255]]=[128,128,128]
        white = np.array([255, 255, 255])
        # gray = np.array([128, 128, 128])
        gray_value = np.random.randint(0, 256)
        gray = np.array([gray_value, gray_value, gray_value])

        mask = np.all(image == white[None, None, :], axis=-1)
        image[mask] = gray # replace white pixels by gray ones

        return image

    for idx in range(10):
        size = random.randint(128, 512)
        aspect_ratio = random.uniform(0.25, 2.5)

        if random.random() < 0.5:
            # Generate square image
            image = generate_image_with_number(size, idx)
        else:
            # Generate rectangular image
            image = generate_image_with_number(size, idx, aspect_ratio)

        # Save the generated image
        plt.imsave(f'image_{idx:02d}.png', image / 255.0)


if __name__ == '__main__':
    generate_random_number_images()