import random
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops
from io import BytesIO

def find_differing_region(fig):
    ax = fig.gca()

    # Save the initial plot as an image in memory
    initial_buffer = BytesIO()
    plt.savefig(initial_buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    initial_buffer.seek(0)

    # Hide the last line
    if ax.get_lines():
        ax.get_lines()[-1].set_visible(False)

    # Save the modified plot as an image in memory
    modified_buffer = BytesIO()
    plt.savefig(modified_buffer, format='png', dpi=300, bbox_inches='tight', pad_inches=0.1)
    modified_buffer.seek(0)

    # Load the images using Pillow from the in-memory buffers
    initial_image = Image.open(initial_buffer).convert('RGB')
    modified_image = Image.open(modified_buffer).convert('RGB')

    # Close the initial buffers as they are no longer needed
    initial_buffer.close()
    modified_buffer.close()

    # Find the differing region
    diff = ImageChops.difference(initial_image, modified_image)

    # Get the bounding box of the differing region
    bbox = diff.getbbox()

    return bbox, diff

def clone_figure(fig): # this is a code to clone a matplotlib figure
    # there are so many parameters missing in there that it is not very useful to use this cloner unless no adjustment to the figure is made

    # Get the axes from the figure
    ax = fig.gca()

    # Create a new figure and axes
    fig_clone, ax_clone = plt.subplots()

    # Copy the lines to the cloned axes
    for line in ax.get_lines():
        ax_clone.plot(line.get_xdata(), line.get_ydata(), label=line.get_label(), color=line.get_color(), linestyle=line.get_linestyle(), linewidth=line.get_linewidth())

    # Copy other properties (e.g., axis limits, labels, title, legend)
    ax_clone.set_xlim(ax.get_xlim())
    ax_clone.set_ylim(ax.get_ylim())
    ax_clone.set_xlabel(ax.get_xlabel())
    ax_clone.set_ylabel(ax.get_ylabel())
    ax_clone.set_title(ax.get_title())
    ax_clone.legend()

    return fig_clone, ax_clone


def plot_image_with_pixels(image, cmap='gray'):
    '''
    Plots an image with white lines surrounding each pixel. NB: This only make sense for small images

    Args:
        image (numpy.ndarray): The input image array.
        cmap (str, optional): The colormap to use for displaying the image. Defaults to 'gray'.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plotted image.

    # Examples:
    #     >>> image = np.random.rand(10, 10)
    #     >>> fig = plot_image_with_pixels(image)
    #     >>> plt.show()
    '''
    # Create a new figure and axes
    fig, ax = plt.subplots()

    # Show the image using imshow()
    ax.imshow(image, cmap=cmap, interpolation=None)

    # Set the tick locations to center the ticks at the pixels
    ax.set_xticks(np.arange(image.shape[1]), minor=False)
    ax.set_yticks(np.arange(image.shape[0]), minor=False)

    # Shift the grid by 0.5 to surround the pixels
    ax.set_xticks(np.arange(image.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(image.shape[0]+1)-0.5, minor=True)

    # If the image is too large, hide the tick labels
    if image.shape[0] > 50:
        ax.set_yticklabels([])

    if image.shape[1] > 50:
        ax.set_xticklabels([])

    # Add the lines around each pixel using grid()
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1)

    # Return the figure object
    return fig


if __name__ == '__main__':

    if True:
        # Generate some data
        x= np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)

        # Create the initial plot with two lines
        fig, ax = plt.subplots()
        line1, = ax.plot(x, y1, label='sin(x)')
        line2, = ax.plot(x, y2, label='cos(x)')
        ax.legend()

        # Display the initial plot
        plt.show()

        # Clone the figure
        fig_clone, ax_clone = clone_figure(fig)

        # Display the cloned plot
        plt.show()

        # Remove one of the lines from the original plot
        line2.remove()

        # Update the legend to reflect the removal of the line
        ax.legend()

        # Display the modified original plot
        plt.show()

    if True:
        # Generate a random image of a random type
        image = random.choice([np.random.rand(10, 10), np.random.rand(100, 10, 3)])

        # Plot the image with white lines surrounding each pixel
        fig = plot_image_with_pixels(image)

        # Show the plot
        plt.show()


    if True:
        from batoolset.img import pop, Img
        img = Img('/E/Sample_images/sample_images_FIJI/AuPbSn40.jpg')
        pop(img)
        # sys.exit(0)




