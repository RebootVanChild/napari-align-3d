"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from aicsimageio import AICSImage, readers
from magicgui import magic_factory
from napari import Viewer

if TYPE_CHECKING:
    pass


# landmarks file has coordinates in microns,
# while Napari takes in transformation matrix in pixels.
# landmarks file in xyz order,
# while Napari takes in transformation matrix in zyx order.
def Matrix_to_napari_affine_input(
    matrix, source_physical_pixel_sizes, target_physical_pixel_sizes
):
    # convert unit: microns to pixels
    support_matrix = [
        [
            source_physical_pixel_sizes.X / target_physical_pixel_sizes.X,
            source_physical_pixel_sizes.Y / target_physical_pixel_sizes.Y,
            source_physical_pixel_sizes.Z / target_physical_pixel_sizes.Z,
            1 / target_physical_pixel_sizes.X,
        ],
        [
            source_physical_pixel_sizes.X / target_physical_pixel_sizes.X,
            source_physical_pixel_sizes.Y / target_physical_pixel_sizes.Y,
            source_physical_pixel_sizes.Z / target_physical_pixel_sizes.Z,
            1 / target_physical_pixel_sizes.Y,
        ],
        [
            source_physical_pixel_sizes.X / target_physical_pixel_sizes.X,
            source_physical_pixel_sizes.Y / target_physical_pixel_sizes.Y,
            source_physical_pixel_sizes.Z / target_physical_pixel_sizes.Z,
            1 / target_physical_pixel_sizes.Z,
        ],
        [1, 1, 1, 1],
    ]
    matrixXYZ = matrix * support_matrix
    # convert order: XYZ to ZYX
    matrixZYX = matrixXYZ
    matrixZYX[:3, :3] = np.rot90(matrixXYZ[:3, :3], 2)
    matrixZYX[:3, 3] = np.flip(matrixXYZ[:3, 3])
    return matrixZYX


# Input: landmarks pairs of source and target points.
# Output: 4x4 rigid body transformation matrix.
def GetRigidMatrixFromLandmarks(
    source_points_landmarks, target_points_landmarks
):
    centroid_source = np.mean(source_points_landmarks, axis=0)
    centroid_target = np.mean(target_points_landmarks, axis=0)
    P = source_points_landmarks - centroid_source
    Q = target_points_landmarks - centroid_target
    M = np.dot(P.T, Q)
    U, W, V = np.linalg.svd(M)
    R = np.dot(V.T, U.T)
    matrix = np.append(
        np.append(
            R,
            np.vstack(-np.dot(R, centroid_source.T) + centroid_target.T),
            axis=1,
        ),
        [[0.0, 0.0, 0.0, 1.0]],
        axis=0,
    )
    print(matrix)
    return matrix


# Input: landmarks pairs of source and target points.
# Output: 4x4 affine transformation matrix.
def GetAffineMatrixFromLandmarks(
    source_points_landmarks, target_points_landmarks
):
    pts_count = len(source_points_landmarks)
    A = np.zeros((pts_count * 3, 12))
    b = np.zeros(pts_count * 3)
    for i in range(pts_count):
        # build A
        A[i * 3][0] = source_points_landmarks[i][0]
        A[i * 3][1] = source_points_landmarks[i][1]
        A[i * 3][2] = source_points_landmarks[i][2]
        A[i * 3][3] = 1
        A[i * 3 + 1][4] = source_points_landmarks[i][0]
        A[i * 3 + 1][5] = source_points_landmarks[i][1]
        A[i * 3 + 1][6] = source_points_landmarks[i][2]
        A[i * 3 + 1][7] = 1
        A[i * 3 + 2][8] = source_points_landmarks[i][0]
        A[i * 3 + 2][9] = source_points_landmarks[i][1]
        A[i * 3 + 2][10] = source_points_landmarks[i][2]
        A[i * 3 + 2][11] = 1
        # build b
        b[i * 3] = target_points_landmarks[i, 0]
        b[i * 3 + 1] = target_points_landmarks[i, 1]
        b[i * 3 + 2] = target_points_landmarks[i, 2]
    x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b.T))
    matrix = np.append(x.reshape(3, 4), [[0.0, 0.0, 0.0, 1.0]], axis=0)
    print(matrix)
    return matrix


@magic_factory(
    call_button="register",
    source_img_file={
        "label": "source image",
        "filter": "*.czi",
        "tooltip": "Select the source image",
    },
    target_img_file={
        "label": "target image",
        "filter": "*.czi",
        "tooltip": "Select the target image",
    },
    landmarks_file={
        "label": "landmarks",
        "filter": "*.csv",
        "tooltip": "Select the landmarks file",
    },
    transformation_type={
        "choices": [
            "affine",
            "rigid body",
        ],
        "tooltip": "Select a transformation type",
    },
    image_channels={
        "choices": [
            "all channels",
            "channel 0",
            "channel 1",
            "channel 2",
            "channel 3",
        ],
        "tooltip": "Select channels to visualize",
    },
)
def example_magic_widget(
    viewer: Viewer,
    source_img_file: Sequence[Path],
    target_img_file: Sequence[Path],
    landmarks_file: Sequence[Path],
    transformation_type: str = "affine",
    image_channels: str = "all channels",
):
    source_img_path = str(source_img_file[0])
    target_img_path = str(target_img_file[0])
    landmarks_path = str(landmarks_file[0])
    # load images and landmarks
    source_img = AICSImage(source_img_path, reader=readers.BioformatsReader)
    target_img = AICSImage(target_img_path, reader=readers.BioformatsReader)
    landmarks = np.loadtxt(
        landmarks_path,
        delimiter=",",
        converters=lambda x: float(eval(x)),
        usecols=range(2, 8),
    )
    source_pts = landmarks[:, 0:3]
    target_pts = landmarks[:, 3:6]
    # load physical pixel sizes (microns per pixel in Z Y X)
    physical_pixel_sizes_source = source_img.physical_pixel_sizes
    physical_pixel_sizes_target = target_img.physical_pixel_sizes
    # calculate Napari input matrix
    if transformation_type == "affine":
        napari_input_matrix = Matrix_to_napari_affine_input(
            GetAffineMatrixFromLandmarks(source_pts, target_pts),
            physical_pixel_sizes_source,
            physical_pixel_sizes_target,
        )
    elif transformation_type == "rigid body":
        napari_input_matrix = Matrix_to_napari_affine_input(
            GetRigidMatrixFromLandmarks(source_pts, target_pts),
            physical_pixel_sizes_source,
            physical_pixel_sizes_target,
        )
    print("transformation in ZYX order:\n", napari_input_matrix)
    if image_channels == "all channels":
        viewer.add_image(
            source_img.data,
            channel_axis=1,
            name=["source_C0", "source_C1", "source_C2", "source_C3"],
            affine=napari_input_matrix,
            blending="additive",
            visible=True,
        )
        viewer.add_image(
            target_img.data,
            channel_axis=1,
            name=["target_C0", "target_C1", "target_C2", "target_C3"],
            blending="additive",
            visible=True,
        )
    else:
        channel_idx_char = image_channels[-1]
        viewer.add_image(
            source_img.get_image_data("ZYX", C=int(channel_idx_char)),
            name="source_C" + channel_idx_char,
            affine=napari_input_matrix,
            colormap="red",
            blending="additive",
            visible=True,
        )
        viewer.add_image(
            target_img.get_image_data("ZYX", C=int(channel_idx_char)),
            name="target_C" + channel_idx_char,
            colormap="green",
            blending="additive",
            visible=True,
        )
    viewer.dims.ndisplay = 3
