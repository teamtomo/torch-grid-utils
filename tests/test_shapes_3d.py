from torch_grid_utils.shapes_3d import sphere
import napari

def _sphere():
    result = sphere(
        radius=15,
        image_shape=50,
        center=None,
        smoothing_radius=5,
    )
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(result.numpy(), name='Sphere')
    napari.run()

if __name__ == '__main__':
    _sphere()
