import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

if __name__ == '__main__':
    folder = os.path.join(os.path.expanduser('~'), 'data', 'model_the_leader')
    to_check = '0_0'

    gt_file = os.path.join(folder, f'{to_check}_ground_truth.pickle')
    eval_file = os.path.join(folder, f'{to_check}_results.pickle')

    ax = plt.figure().add_subplot(projection='3d')

    for file, label in [(gt_file, 'Ground Truth'), (eval_file, 'Results')]:
        with open(file, 'rb') as fh:
            data = pickle.load(fh)

        kwargs = {}
        if label == 'Ground Truth':
            kwargs = {'linestyle': 'dashed'}

        ax.plot(*data['leader'].T, color='blue', **kwargs)
        for sb in data['side_branches']:
            ax.plot(*sb.T, color='green', **kwargs)
    set_axes_equal(ax)
    plt.show()

