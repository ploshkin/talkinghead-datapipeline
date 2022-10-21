def test_pytorch3d_imports() -> None:
    # Imports that used in `dpl/rendering/renderer.py`
    from pytorch3d.structures import Meshes
    from pytorch3d.io import load_obj
    from pytorch3d.renderer.mesh import rasterize_meshes
    