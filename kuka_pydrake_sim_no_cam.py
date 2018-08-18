# -*- coding: utf8 -*-

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np

import pydrake
from pydrake.all import (
    DiagramBuilder,
    RgbdCamera,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    RungeKutta2Integrator,
    Shape,
    SignalLogger,
    Simulator,
)

from underactuated.meshcat_rigid_body_visualizer import (
    MeshcatRigidBodyVisualizer)

from pydrake.trajectories import (
    PiecewisePolynomial
)

from pydrake.math import (
    RollPitchYaw
)

import kuka_controllers
import kuka_ik
import kuka_utils

if __name__ == "__main__":
    np.set_printoptions(precision=5, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-T", "--duration",
                        type=float,
                        help="Duration to run sim.",
                        default=4.0)
    parser.add_argument("--test",
                        action="store_true",
                        help="Help out CI by launching a meshcat server for "
                             "the duration of the test.")
    args = parser.parse_args()

    meshcat_server_p = None
    if args.test:
        print "Spawning"
        import subprocess
        meshcat_server_p = subprocess.Popen(["meshcat-server"])
    else:
        print "Warning: if you have not yet run meshcat-server in another " \
              "terminal, this will hang."

    # Construct the robot and its environment
    rbt = RigidBodyTree()
    kuka_utils.setup_kuka(rbt)

    # Set up a visualizer for the robot
    pbrv = MeshcatRigidBodyVisualizer(rbt, draw_timestep=0.01)
    # (wait while the visualizer warms up and loads in the models)
    time.sleep(2.0)

    # Plan a robot motion to maneuver from the initial posture
    # to a posture that we know should grab the object.
    # (Grasp planning is left as an exercise :))
    q0 = rbt.getZeroConfiguration()
    # qtraj, info = kuka_ik.plan_grasping_trajectory(
    #     rbt,
    #     q0=q0,
    #     target_reach_pose=np.array([0.6, 0., 1.0, -0.75, 0., -1.57]),
    #     target_grasp_pose=np.array([0.8, 0., 0.9, -0.75, 0., -1.57]),
    #     n_knots=20,
    #     reach_time=1.5,
    #     grasp_time=2.0)

    #--------------------------------------------------------------
    q1 = np.array([[ 0.2793],
       [0.6824],
       [-0.0456],
       [-1.4918],
       [0.0754],
       [0.9042],
       [0.5961],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.]])

    q2 = np.array([[-0.1456],
       [-0.6498],
       [0.1090],
       [-1.5984],
       [0.0794],
       [1.5141],
       [0.4069],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.]])

    q3 = np.zeros(rbt.get_num_positions())
    q3[0:7] = np.array([0.92757,  1.06807, -0.77407, -1.90249, -0.45508, -0.40019,
        1.2063])
    q3.resize((rbt.get_num_positions(), 1))

    target_reach_pose = np.array([0.6, 0.15, 0.95, -0.75, 0., -1.57])
    q_pre_grasp, info = kuka_ik.plan_grasping_configuration(rbt, q1.flatten(), q0, target_reach_pose)

    kinsol = rbt.doKinematics(q_pre_grasp)
    idx_ee = 13
    H_WE = rbt.CalcBodyPoseInWorldFrame(kinsol, rbt.get_body(idx_ee))
    R_WEr = H_WE[0:3, 0:3]

    qtraj = PiecewisePolynomial.FirstOrderHold([0, 1.5], np.hstack((q2, q_pre_grasp.reshape(q2.shape))))
    #---------------------------------------------------------------

    # Make our RBT into a plant for simulation
    rbplant = RigidBodyPlant(rbt)
    rbplant.set_name("Rigid Body Plant")

    # Build up our simulation by spawning controllers and loggers
    # and connecting them to our plant.
    builder = DiagramBuilder()
    # The diagram takes ownership of all systems
    # placed into it.
    rbplant_sys = builder.AddSystem(rbplant)

    # Create a high-level state machine to guide the robot
    # motion...
    manip_state_machine = builder.AddSystem(
        kuka_controllers.ManipStateMachine(rbt, rbplant_sys, qtraj, R_WEr))
    builder.Connect(rbplant_sys.state_output_port(),
                    manip_state_machine.robot_state_input_port)
    builder.Connect(rbplant_sys.contact_results_output_port(),
                    manip_state_machine.contact_result_input_port)

    # And spawn the controller that drives the Kuka to its
    # desired posture.
    kuka_controller = builder.AddSystem(
        kuka_controllers.KukaController(rbt, rbplant_sys))
    builder.Connect(rbplant_sys.state_output_port(),
                    kuka_controller.robot_state_input_port)
    builder.Connect(manip_state_machine.kuka_plan_output_port,
                    kuka_controller.plan_input_port)
    builder.Connect(kuka_controller.get_output_port(0),
                    rbplant_sys.get_input_port(0))

    # Same for the hand
    hand_controller = builder.AddSystem(
        kuka_controllers.HandController(rbt, rbplant_sys))
    builder.Connect(rbplant_sys.state_output_port(),
                    hand_controller.robot_state_input_port)
    builder.Connect(manip_state_machine.hand_setpoint_output_port,
                    hand_controller.setpoint_input_port)
    builder.Connect(hand_controller.get_output_port(0),
                    rbplant_sys.get_input_port(1))

    # Hook up the visualizer we created earlier.
    visualizer = builder.AddSystem(pbrv)
    builder.Connect(rbplant_sys.state_output_port(),
                    visualizer.get_input_port(0))

    # Hook up contact logger
    contact_logger = builder.AddSystem(kuka_utils.ContactLogger(rbplant_sys))
    contact_logger._DeclarePeriodicPublish(1. / 100, 0.0)
    builder.Connect(rbplant_sys.contact_results_output_port(), contact_logger.get_input_port(0))

    # Hook up loggers for the robot state, the robot setpoints,
    # and the torque inputs.
    def log_output(output_port, rate):
        logger = builder.AddSystem(SignalLogger(output_port.size()))
        logger._DeclarePeriodicPublish(1. / rate, 0.0)
        builder.Connect(output_port, logger.get_input_port(0))
        return logger
    state_log = log_output(rbplant_sys.get_output_port(0), 60.)
    kuka_control_log = log_output(
        kuka_controller.get_output_port(0), 60.)

    # Done! Compile it all together and visualize it.
    diagram = builder.Build()
    kuka_utils.render_system_with_graphviz(diagram, "view.gv")

    # Create a simulator for it.
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(0.0)
    # Simulator time steps will be very small, so don't
    # force the rest of the system to update every single time.
    simulator.set_publish_every_time_step(False)

    # The simulator simulates forward from a given Context,
    # so we adjust the simulator's initial Context to set up
    # the initial state.
    state = simulator.get_mutable_context().\
        get_mutable_continuous_state_vector()
    initial_state = np.zeros(state.size())
    initial_state[0:q2.shape[0]] = q2.flatten()
    state.SetFromVector(initial_state)

    # From iiwa_wsg_simulation.cc:
    # When using the default RK3 integrator, the simulation stops
    # advancing once the gripper grasps the box.  Grasping makes the
    # problem computationally stiff, which brings the default RK3
    # integrator to its knees.
    timestep = 0.0002
    simulator.reset_integrator(
        RungeKutta2Integrator(diagram, timestep,
                              simulator.get_mutable_context()))

    # This kicks off simulation. Most of the run time will be spent
    # in this call.
    # simulator.StepTo(args.duration)
    simulator.StepTo(7.0)

    x_final = state.CopyToVector()
    print("desired EE orientation: ", R_WEr)
    print("final EE pose @ qf: ", manip_state_machine.CalcEEPoseInWorldFrame(x_final[0:rbt.get_num_positions()]))
    print("Final state: ", x_final)

    if args.test is not True:
        # Do some plotting to show off accessing signal logger data.
        nq = rbt.get_num_positions()
        plt.figure()
        plt.subplot(3, 1, 1)
        dims_to_draw = range(7)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 7)))
        for i in dims_to_draw:
            colorthis = next(color)
            plt.plot(state_log.sample_times(),
                     state_log.data()[i, :],
                     color=colorthis,
                     linestyle='solid',
                     label="q[%d]" % i)
        plt.ylabel("m")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.subplot(3, 1, 2)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 7)))
        for i in dims_to_draw:
            colorthis = next(color)
            plt.plot(state_log.sample_times(),
                     state_log.data()[nq + i, :],
                     color=colorthis,
                     linestyle='solid',
                     label="v[%d]" % i)
        plt.ylabel("m/s")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

        plt.subplot(3, 1, 3)
        color = iter(plt.cm.rainbow(np.linspace(0, 1, 7)))
        for i in dims_to_draw:
            colorthis = next(color)
            plt.plot(kuka_control_log.sample_times(),
                     kuka_control_log.data()[i, :],
                     color=colorthis,
                     linestyle=':',
                     label="u[%d]" % i)
        plt.xlabel("t")
        plt.ylabel("N/m")
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    if meshcat_server_p is not None:
        meshcat_server_p.kill()
        meshcat_server_p.wait()

