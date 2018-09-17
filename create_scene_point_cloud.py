import numpy as np
from matplotlib import cm

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.rigid_body_tree import (
    AddModelInstanceFromUrdfFile,
    FloatingBaseType,
    RigidBodyTree,
    RigidBodyFrame,
    )
from pydrake.systems.framework import (
    BasicVector,
    )
from pydrake.systems.sensors import (
    CameraInfo,
    RgbdCamera,
    )

import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g

import kuka_utils


# Create tree describing scene.
urdf_path = FindResourceOrThrow(
    "drake/examples/kuka_iiwa_arm/models/objects/black_box.urdf")
tree = RigidBodyTree()
kuka_utils.setup_kuka(tree)

# - Add frames for camera fixture.
frames = (
    RigidBodyFrame(
        name="rgbd camera frame 1",
        body=tree.world(),
        xyz=[0.8, 0.6, 1.5],  # Ensure that the box is within range.
        rpy=[0, np.pi / 3, -np.pi / 2]),
    RigidBodyFrame(
        name="rgbd camera frame 2",
        body=tree.world(),
        xyz=[0.8, -0.6, 1.5],  # Ensure that the box is within range.
        rpy=[0, np.pi / 3, np.pi / 2]),
    RigidBodyFrame(
        name="rgbd camera frame 3",
        body=tree.world(),
        xyz=[1.2, 0, 1.5],  # Ensure that the box is within range.
        rpy=[0, np.pi / 3, np.pi]),
    RigidBodyFrame(
        name="rgbd camera frame 4",
        body=tree.world(),
        xyz=[0.4, 0, 1.5],  # Ensure that the box is within range.
        rpy=[0, np.pi / 3, 0]),
)


cameras = []

for i, frame in enumerate(frames):
    tree.addFrame(frame)
    # Create camera.
    cameras.append(RgbdCamera(
        name="camera%d"%i, tree=tree, frame=frame,
        z_near=0.1, z_far=5.0,
        fov_y=np.pi / 3, show_window=False))

# - Describe state.
x = np.zeros(tree.get_num_positions() + tree.get_num_velocities())
kinsol = tree.doKinematics(x[:tree.get_num_positions()])

# Allocate context and render.
points_in_world_frame = np.zeros((3, 0))
colors = np.zeros((3, 0))
for camera in cameras:
    context = camera.CreateDefaultContext()
    context.FixInputPort(0, BasicVector(x))
    output = camera.AllocateOutput()
    camera.CalcOutput(context, output)

    # Get images from computed output.
    color_index = camera.color_image_output_port().get_index()
    color_image = output.get_data(color_index).get_value()
    color_array = color_image.data

    depth_index = camera.depth_image_output_port().get_index()
    depth_image = output.get_data(depth_index).get_value()
    depth_array = np.squeeze(depth_image.data)

    w, h = depth_array.shape

    # Convert depth image to point cloud, with +z being
    # camera "forward"

    tree = camera.tree()
    Kinv = np.linalg.inv(
        camera.depth_camera_info().intrinsic_matrix())
    U, V = np.meshgrid(np.arange(h), np.arange(w))
    points_in_camera_frame = np.vstack([
        U.flatten(),
        V.flatten(),
        np.ones(w * h)])
    points_in_camera_frame = Kinv.dot(points_in_camera_frame) * \
                             depth_array.flatten()

    # The depth camera has some offset from the camera's root frame,
    # so take than into account.
    pose_mat = camera.depth_camera_optical_pose().matrix()
    points_in_camera_frame = pose_mat[0:3, 0:3].dot(points_in_camera_frame)
    points_in_camera_frame += np.tile(pose_mat[0:3, 3], [w * h, 1]).T


    points_in_world_frame = np.hstack((points_in_world_frame,
                                       tree.transformPoints(
                                           kinsol,
                                           points_in_camera_frame,
                                           camera.frame().get_frame_index(),
                                           0)))


prefix="RBCameraViz"
vis = meshcat.Visualizer(zmq_url="tcp://127.0.0.1:6000")
vis.delete()
# vis[prefix].delete()
# Color points according to their normalized height
min_height = 0.0
max_height = 2.0
colors = cm.jet((points_in_world_frame[2, :] - min_height) / (max_height - min_height)).T[0:3, :]
vis[prefix]["points"].set_object(
    g.PointCloud(position=points_in_world_frame,
                 color=colors,
                 size=0.002))