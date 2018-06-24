
import os
import numpy as np
import matplotlib.pyplot as plt

import pydrake
from pydrake.all import (
    DiagramBuilder,
    RigidBodyFrame,
    RigidBodyPlant,
    RigidBodyTree,
    AddModelInstanceFromUrdfFile,
    FloatingBaseType,
    PortDataType,
    BasicVector,
    RungeKutta2Integrator,
    Shape,
    SignalLogger,
    Simulator,
    LeafSystem,
    AddFlatTerrainToWorld,
    ConstantVectorSource,
    MathematicalProgram,
)

from underactuated.meshcat_rigid_body_visualizer import (
    MeshcatRigidBodyVisualizer)

from pydrake.trajectories import (
    PiecewisePolynomial
)
#%%
class ContactLogger(LeafSystem):
    ''' Logs contact force history, using
        the rigid body plant contact result
        output port.

        Stores sample times, accessible via
        sample_times(), and contact results for
        each sample time, accessible as a list
        from method data().

        Every contact result is a list of tuples,
        one tuple for each contact,
        where each tuple contains (id_1, id_2, r, f):
            id_1 = the ID of element #1 in collision
            id_2 = the ID of element #2 in collision
            r = the contact location, in world frame
            f = the contact force, in world frame'''

    def __init__(self, plant):
        LeafSystem.__init__(self)

        self._data = []
        self._sample_times = np.empty((0, 1))
        self.shut_up = False
        # Contact results
        self._DeclareInputPort(PortDataType.kAbstractValued,
                               plant.contact_results_output_port().size())

    def data(self):
        return self._data

    def sample_times(self):
        return self._sample_times

    def _DoPublish(self, context, events):
        contact_results = self.EvalAbstractInput(context, 0).get_value()
        self._sample_times = np.vstack([self._sample_times, [context.get_time()]])

        this_contact_info = []
        for contact_i in range(contact_results.get_num_contacts()):
            # if contact_i >= self.n_cf:
            #     if not self.shut_up:
            #         print "More contacts than expected (the # of grasp points). " \
            #               "Dropping some! Your fingertips probably touched each other."
            #         self.shut_up = True
            #     break
            # Cludgy -- would rather keep things as objects.
            # But I need to work out how to deepcopy those objects.
            # (Need to bind their various constructive methods)
            contact_info = contact_results.get_contact_info(contact_i)
            contact_force = contact_info.get_resultant_force()
            this_contact_info.append([
                contact_info.get_element_id_1(),
                contact_info.get_element_id_2(),
                contact_force.get_application_point(),
                contact_force.get_force(),
                contact_results.get_generalized_contact_force()
            ])
        self._data.append(this_contact_info)


def PrintRbtInfo(tree):
    n_positions = tree.get_num_positions()
    for i in range(n_positions):
        print i, tree.get_position_name(i)

    print "-----------------"

    n_bodies = tree.get_num_bodies()
    for i in range(n_bodies):
        print i, tree.getBodyOrFrameName(i)

    print "Number of actuators:", tree.get_num_actuators()


# joint trajectory
t = np.array([0, 2, 4])
q_knots = np.array([[0.5, 1, 1.5], [0.05, 0.04, 0.05]])
qtraj = PiecewisePolynomial.Cubic(t, q_knots, [0,0], [0,0])
qtraj_d = qtraj.derivative(1)
qtraj_dd = qtraj_d.derivative(1)


class JointSpaceController(LeafSystem):
    def __init__(self, tree, control_period=0.005):
        LeafSystem.__init__(self)

        self.nq = tree.get_num_positions()
        self.na = tree.get_num_actuators()
        self.B_inv = np.linalg.inv(tree.B)
        self.tree = tree

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued, \
                                   tree.get_num_positions() +
                                   tree.get_num_velocities())
        self._DeclareVectorOutputPort(BasicVector(self.na), self._DoCalcVectorOutput)
        self._DeclareDiscreteState(self.na)  # state of the controller system is u
        self._DeclarePeriodicDiscreteUpdate(period_sec=control_period)  # update u every h seconds.


    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        new_control_input = discrete_state.get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(context, self.robot_state_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]

        t = context.get_time()
        q_ref = qtraj.value(t).flatten()
        qd_ref = qtraj_d.value(t).flatten()
        qdd_ref = qtraj_dd.value(t).flatten()

        err_q = q_ref - q
        err_v = qd_ref - v

        qdd_des = qdd_ref + 1000* err_q + 100.*err_v
        kinsol = self.tree.doKinematics(q, v)
        lhs = self.tree.inverseDynamics(kinsol, external_wrenches={}, vd=qdd_des)
        new_u = self.B_inv.dot(lhs)
        new_control_input[:] = new_u

    def _DoCalcVectorOutput(self, context, y_data):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = control_output


class RobotController(LeafSystem):
    def __init__(self, tree, control_period=0.005):
        LeafSystem.__init__(self)

        self.nq = tree.get_num_positions()
        self.na = tree.get_num_actuators()
        self.B_inv = np.linalg.inv(tree.B)
        self.tree = tree

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued, \
                                   tree.get_num_positions() +
                                   tree.get_num_velocities())
        self._DeclareVectorOutputPort(BasicVector(self.na), self._DoCalcVectorOutput)
        self._DeclareDiscreteState(self.na)  # state of the controller system is u
        self._DeclarePeriodicDiscreteUpdate(period_sec=control_period)  # update u every h seconds.


    def _DoCalcVectorOutput(self, context, y_data):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = control_output


class TaskSpaceController(RobotController):
    def __init__(self, tree, control_period=0.005):
        RobotController.__init__(self, tree, control_period)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        new_control_input = discrete_state.get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(context, self.robot_state_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]
        kinsol = self.tree.doKinematics(q, v)
        idx_base = 1
        idx_ee = 5
        J_ee, v_indices = self.tree.geometricJacobian(kinsol, idx_base, idx_ee, idx_base)
        T_ee = self.tree.CalcBodyPoseInWorldFrame(kinsol, tree.get_body(idx_ee))
        J_ee = J_ee[4:6, :] # q to x and y velocity in world frame.
        x_ee = T_ee[1:3, 3]
        xd_ee = J_ee.dot(v)
        J_eeDotTimesV = self.tree.geometricJacobianDotTimesV(kinsol, idx_base, idx_ee, idx_base)


        t = context.get_time()
        x_ee_des = qtraj.value(t).flatten()
        xd_ee_des = qtraj_d.value(t).flatten()
        xdd_ee_des = qtraj_dd.value(t).flatten()

        err_x_ee = x_ee_des - x_ee
        err_xd_ee = xd_ee_des - xd_ee

        qdd_des = np.linalg.solve(J_ee, xdd_ee_des - J_eeDotTimesV[4:6])

        err_q = np.linalg.solve(J_ee, err_x_ee)
        err_qd = np.linalg.solve(J_ee, err_xd_ee)

        qdd_ref = qdd_des + 50.*err_q + 10*err_qd

        lhs = self.tree.inverseDynamics(kinsol, external_wrenches={}, vd=qdd_ref)
        new_u = self.B_inv.dot(lhs)
        new_control_input[:] = new_u


class QpInverseDynamicsController(RobotController):
    def __init__(self, tree, control_period=0.005):
        RobotController.__init__(self, tree, control_period)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        new_control_input = discrete_state.get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(context, self.robot_state_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]
        kinsol = self.tree.doKinematics(q, v)
        idx_base = 1
        idx_ee = 5
        J_ee, v_indices = self.tree.geometricJacobian(kinsol, idx_base, idx_ee, idx_base)
        T_ee = self.tree.CalcBodyPoseInWorldFrame(kinsol, tree.get_body(idx_ee))
        J_ee = J_ee[4:6, :] # q to x and y velocity in world frame.
        x_ee = T_ee[1:3, 3]
        xd_ee = J_ee.dot(v)
        J_eeDotTimesV = self.tree.geometricJacobianDotTimesV(kinsol, idx_base, idx_ee, idx_base)


        t = context.get_time()
        x_ee_des = qtraj.value(t).flatten()
        xd_ee_des = qtraj_d.value(t).flatten()
        xdd_ee_des = qtraj_dd.value(t).flatten()

        err_x_ee = x_ee_des - x_ee
        err_xd_ee = xd_ee_des - xd_ee

        qdd_des = np.linalg.solve(J_ee, xdd_ee_des - J_eeDotTimesV[4:6])

        err_q = np.linalg.solve(J_ee, err_x_ee)
        err_qd = np.linalg.solve(J_ee, err_xd_ee)

        qdd_ref = qdd_des + 50.*err_q + 10*err_qd

        lhs = self.tree.inverseDynamics(kinsol, external_wrenches={}, vd=qdd_ref)
        new_u = self.B_inv.dot(lhs)
        new_control_input[:] = new_u


if __name__ == "__main__":
#%%
    tree = RigidBodyTree()
    arm_urdf_path = os.path.join(
        pydrake.getDrakePath(),
        "manipulation", "models", "two_link_arm.urdf")

    AddModelInstanceFromUrdfFile(arm_urdf_path, FloatingBaseType.kFixed, None, tree)

    # AddFlatTerrainToWorld(tree)

    meshcat_vis = MeshcatRigidBodyVisualizer(tree, draw_timestep=0.01)
    alpha = np.arccos(0.25)
    q0 = np.array([alpha, -alpha*2])
    meshcat_vis.draw(q0)
#%% setup simulation diagram
    # q0 = qtraj.value(0).flatten()
    t_final = 5.0
    plant = RigidBodyPlant(tree)
    plant.set_name("rigid_body_plant")

    builder = DiagramBuilder()
    builder.AddSystem(plant)
    controller = builder.AddSystem(TaskSpaceController(tree))
    # controller = builder.AddSystem(ConstantVectorSource([0,0]))
    visualizer = builder.AddSystem(meshcat_vis)
    state_logger = builder.AddSystem(SignalLogger(plant.state_output_port().size()))
    state_d_logger = builder.AddSystem(SignalLogger(plant.state_derivative_output_port().size()))
    contact_logger = builder.AddSystem(ContactLogger(plant))
    torque_logger = builder.AddSystem(SignalLogger(plant.torque_output_port().size()))

    builder.Connect(controller.get_output_port(0), plant.get_input_port(0))
    builder.Connect(plant.state_output_port(), controller.robot_state_input_port)
    builder.Connect(plant.state_output_port(), visualizer.get_input_port(0))
    builder.Connect(plant.state_output_port(), state_logger.get_input_port(0))
    builder.Connect(plant.contact_results_output_port(),
                    contact_logger.get_input_port(0))
    builder.Connect(plant.torque_output_port(), torque_logger.get_input_port(0))
    builder.Connect(plant.state_derivative_output_port(), state_d_logger.get_input_port(0))

    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(2.0)

    state = simulator.get_mutable_context(). \
        get_mutable_continuous_state_vector()
    initial_state = np.zeros(state.size())
    initial_state[0:q0.shape[0]] = q0
    state.SetFromVector(initial_state)

    # timestep = 0.0001
    # simulator.reset_integrator(
    #     RungeKutta2Integrator(diagram, timestep,
    #                           simulator.get_mutable_context()))

    simulator.StepTo(t_final)

#%% processing logs
    x = state_logger.data()
    t = state_logger.sample_times()

    idx_ee = 5
    x_ee = np.zeros((2, t.size))
    tau = torque_logger.data()
    xd = state_d_logger.data()
    tau_d = np.zeros(t.size)
    tau_contact_norm = np.zeros(t.size)
    f_contact_norm = np.zeros(t.size)
    for i, xi in enumerate(x.T):
        q = xi[0:2]
        v = xi[2:4]
        kinsol = tree.doKinematics(q, v)
        T_ee = tree.CalcBodyPoseInWorldFrame(kinsol, tree.get_body(idx_ee))
        x_ee[:,i] = T_ee[1:3, 3]

        vd = xd[2:4, i]
        tau_contact = np.zeros(2)
        f_contact = np.zeros(3)
        if contact_logger.data()[i]:
            tau_contact = contact_logger.data()[i][0][-1]
            f_contact = contact_logger.data()[i][0][-2]
        tau_contact_norm[i] = np.linalg.norm(tau_contact)
        tau_d[i] = np.linalg.norm(tau[:,i] + tau_contact - tree.inverseDynamics(kinsol, {}, vd))
        f_contact_norm[i] = np.linalg.norm(f_contact)
    # plt.plot(x_ee[0,1:], x_ee[1,1:])
    plt.plot(t, tau_d)
    plt.show()
    plt.plot(t, tau_contact_norm)
    plt.show()
    plt.plot(t, f_contact_norm)
    plt.show()
