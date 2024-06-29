from torch_grid_utils import coordinate_grid, fftfreq_grid
from matplotlib import pyplot as plt
from cmap import Colormap


# simple coordinate grid
fig, axs = plt.subplots(nrows=1, ncols=2)

grid = coordinate_grid(
    image_shape=(28, 28),
    origin=None,
    norm=False,
)

axs[0].imshow(grid[..., 0].numpy(), interpolation='nearest', cmap='gray', origin='lower')
axs[1].imshow(grid[..., 1].numpy(), interpolation='nearest', cmap='gray', origin='lower')
axs[0].axis('off')
axs[1].axis('off')
axs[0].set(title='grid[:, :, 0]', ylabel='height', xlabel='width')
axs[1].set(title='grid[:, :, 1]', xlabel='width')
plt.tight_layout(pad=0.3)
plt.savefig('docs/assets/coordinate_grid_simple.png', dpi=600)


# coordinate grid with origin at (14, 14)
fig, axs = plt.subplots(nrows=1, ncols=2)

grid = coordinate_grid(
    image_shape=(28, 28),
    origin=(14, 14),
    norm=False,
)
cmap = Colormap('cmocean:balance').to_matplotlib()
axs[0].imshow(grid[..., 0].numpy(), interpolation='nearest', cmap=cmap)
im = axs[1].imshow(grid[..., 1].numpy(), interpolation='nearest', cmap=cmap)
axs[0].axis('off')
axs[1].axis('off')
axs[0].set(title='grid[:, :, 0]')
axs[1].set(title='grid[:, :, 1]')
plt.tight_layout(pad=0.3)
plt.savefig('docs/assets/coordinate_grid_centered.png', dpi=600)


# coordinate grid with origin at (14, 14), normed
fig, ax = plt.subplots(nrows=1, ncols=1)

grid = coordinate_grid(
    image_shape=(28, 28),
    origin=(14, 14),
    norm=True,
)
ax.imshow(grid.numpy(), interpolation='nearest', cmap='gray')
ax.axis('off')

plt.tight_layout(pad=0.1)
plt.savefig('docs/assets/coordinate_grid_centered_normed.png', dpi=600)
