import numpy as np
import torch
import torch.nn.functional as F
import math
from collections import OrderedDict
import os
from scipy.ndimage import morphology
from skimage.io import imsave
import cv2


def get_radial_uv(uv_size: int, batch_size: int, device: torch.device) -> torch.Tensor:
    albedo = torch.full(
        [batch_size, 3, uv_size, uv_size],
        fill_value=0.5,  # gray
        device=device,
    )
    grid = torch.linspace(0, 1, uv_size, device=device)
    albedo[:, 0] = grid.unsqueeze(0).repeat(uv_size, 1)
    albedo[:, 1] = grid.flip(0).unsqueeze(1).repeat(1, uv_size)
    return albedo


def _torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)


def _fix_image(image):
    # Taken from EMOCA repo.
    if image.max() < 30.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def to_image(tensor: torch.Tensor) -> np.ndarray:
    return _fix_image(_torch_img_to_np(tensor))


# ---------------------------- process/generate vertices, normals, faces
# Generates faces for a UV-mapped mesh. Each quadruple of neighboring pixels (2x2) is turned into two triangles
def generate_triangles(h, w, mask=None):
    # quad layout:
    # 0 1 ... w-1
    # w w+1
    # .
    # w*h
    triangles = []
    margin = 0
    for x in range(margin, w - 1 - margin):
        for y in range(margin, h - 1 - margin):
            triangle0 = [y * w + x, y * w + x + 1, (y + 1) * w + x]
            triangle1 = [y * w + x + 1, (y + 1) * w + x + 1, (y + 1) * w + x]
            triangles.append(triangle0)
            triangles.append(triangle1)
    triangles = np.array(triangles)
    triangles = triangles[:, [0, 2, 1]]
    return triangles


# copy from https://github.com/daniilidis-group/neural_renderer/blob/master/neural_renderer/vertices_to_faces.py
def face_vertices(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]


def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert vertices.ndimension() == 3
    assert faces.ndimension() == 3
    assert vertices.shape[0] == faces.shape[0]
    assert vertices.shape[2] == 3
    assert faces.shape[2] == 3
    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = (
        faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    )  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1].long(),
        torch.cross(
            vertices_faces[:, 2] - vertices_faces[:, 1],
            vertices_faces[:, 0] - vertices_faces[:, 1],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 2].long(),
        torch.cross(
            vertices_faces[:, 0] - vertices_faces[:, 2],
            vertices_faces[:, 1] - vertices_faces[:, 2],
        ),
    )
    normals.index_add_(
        0,
        faces[:, 0].long(),
        torch.cross(
            vertices_faces[:, 1] - vertices_faces[:, 0],
            vertices_faces[:, 2] - vertices_faces[:, 0],
        ),
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return normals


def batch_orth_proj(X, camera):
    """
    X is N x num_points x 3
    """
    camera = camera.clone().view(-1, 1, 3)
    X_trans = X[:, :, :2] + camera[:, :, 1:]
    X_trans = torch.cat([X_trans, X[:, :, 2:]], 2)
    # shape = X_trans.shape
    # Xn = (camera[:, :, 0] * X_trans.view(shape[0], -1)).view(shape)
    Xn = camera[:, :, 0:1] * X_trans
    return Xn


####################
def load_local_mask(image_size=256, mode="bbx"):
    if mode == "bbx":
        # UV space face attributes bbx in size 2048 (l r t b)
        face = np.array([400, 1648, 400, 1648])

        forehead = np.array([550, 1498, 430, 700 + 50])
        eye_nose = np.array([490, 1558, 700, 1050 + 50])
        mouth = np.array([574, 1474, 1050, 1550])
        ratio = image_size / 2048.0
        face = (face * ratio).astype(np.int)
        forehead = (forehead * ratio).astype(np.int)
        eye_nose = (eye_nose * ratio).astype(np.int)
        mouth = (mouth * ratio).astype(np.int)
        regional_mask = np.array([face, forehead, eye_nose, mouth])

    return regional_mask


def texture2patch(texture, regional_mask, new_size=None):
    patch_list = []
    for pi in range(len(regional_mask)):
        patch = texture[
            :,
            :,
            regional_mask[pi][2] : regional_mask[pi][3],
            regional_mask[pi][0] : regional_mask[pi][1],
        ]
        if new_size is not None:
            patch = F.interpolate(patch, [new_size, new_size], mode="bilinear")
        patch_list.append(patch)
    return patch_list
