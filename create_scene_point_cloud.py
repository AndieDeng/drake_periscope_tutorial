import numpy as np
from matplotlib import cm
import os
import pydrake
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
from pydrake.all import (
    AddFlatTerrainToWorld,
    AddModelInstancesFromSdfString,
    AddModelInstanceFromUrdfFile,
    AddModelInstanceFromUrdfStringSearchingInRosPackages,
    FloatingBaseType,
    LeafSystem,
    PortDataType,
    RigidBodyFrame,
    RollPitchYaw,
    RotationMatrix
)
import meshcat
import meshcat.transformations as tf
import meshcat.geometry as g

import kuka_utils
from underactuated import MeshcatRigidBodyVisualizer

# Create tree describing scene.
object_file_name = "021_bleach_clenser.urdf"
# object_file_name = "004_sugar_box.urdf"
# object_file_name = "apple.urdf"
tree = RigidBodyTree()
#kuka_utils.setup_kuka(tree, object_file_name)
building_sdf_path = os.path.join(
    os.getcwd(), "models", "cmu_building.sdf")
uav_urdf_path = os.path.join(
    os.getcwd(), "models", "021_bleach_clenser.urdf")
AddFlatTerrainToWorld(tree)
table_frame_robot = RigidBodyFrame(
    "table_frame_robot", tree.world(),
    [-1.5, -1, 0], [0, 0, 0])
AddModelInstancesFromSdfString(
    open(building_sdf_path).read(), FloatingBaseType.kFixed,
    table_frame_robot, tree)

# Add UAV model
uav_frame = RigidBodyFrame(
    "uav_frame", tree.world(),
    [0, 0, 0], [0, 0, 0])
AddModelInstanceFromUrdfFile(
    uav_urdf_path, FloatingBaseType.kRollPitchYaw,
    uav_frame, tree)
# - Add frames for camera ficontextxture.
frames = (
    RigidBodyFrame(
        name="rgbd camera frame 1",
        body=tree.FindBody("base_link_apple"),
        xyz=[0,0,0], # [0.8, 0.6, 1.5],  # Ensure that the box is within range.
        rpy=[0,0,0]),# [0, np.pi / 3, -np.pi / 2]),
    # RigidBodyFrame(
    #     name="rgbd camera frame 2",
    #     body=tree.world(),
    #     xyz=[0.8, -0.6, 1.5],  # Ensure that the box is within range.
    #     rpy=[0, np.pi / 3, np.pi / 2]),
    # RigidBodyFrame(
    #     name="rgbd camera frame 3",
    #     body=tree.world(),
    #     xyz=[1.2, 0, 1.5],  # Ensure that the box is within range.
    #     rpy=[0, np.pi / 3, np.pi]),
    # RigidBodyFrame(
    #     name="rgbd camera frame 4",
    #     body=tree.world(),
    #     xyz=[0.4, 0, 1.5],  # Ensure that the box is within range.
    #     rpy=[0, np.pi / 3, 0]),
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
x[0] = 0
x[1] = 0
x[2] = 1
x[4] = np.pi/4
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


tree_viz = MeshcatRigidBodyVisualizer(rbtree=tree)
context_viz = tree_viz.CreateDefaultContext()
context_viz.FixInputPort(0, BasicVector(x))
tree_viz.draw(context_viz)


np.save("scene_point_cloud", points_in_world_frame.T)


