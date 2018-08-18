# -*- coding: utf8 -*-

import numpy as np
import pydrake
from pydrake.all import (
    BasicVector,
    LeafSystem,
    PortDataType,
    MathematicalProgram,
    AbstractValue,
)
import kuka_utils
from pydrake.trajectories import (
    PiecewisePolynomial
)

kuka_controlled_joint_names = [
    "iiwa_joint_1",
    "iiwa_joint_2",
    "iiwa_joint_3",
    "iiwa_joint_4",
    "iiwa_joint_5",
    "iiwa_joint_6",
    "iiwa_joint_7",
]

plan_types = [
    "JointSpacePlan",
    "TaskSpacePlan",
]

idx_world = 0
idx_ee = 13


class Plan:
    def __init__(self,
                 type = None,
                 trajectory = None,
                 start_time = None,
                 R_WE_ref = None):
        self.type = type
        self.traj = trajectory
        if trajectory is not None:
            self.traj_d = trajectory.derivative(1)
        self.t_start = start_time
        self.R_WE_ref = R_WE_ref


class KukaController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.005,
                 print_period=0.5):
        LeafSystem.__init__(self)
        self.set_name("Kuka Controller")

        self.controlled_inds, _ = kuka_utils.extract_position_indices(
            rbt, kuka_controlled_joint_names)
        # Extract the full-rank bit of B, and verify that it's full rank
        self.nq_reduced = len(self.controlled_inds)
        self.B = np.empty((self.nq_reduced, self.nq_reduced))
        for k in range(self.nq_reduced):
            for l in range(self.nq_reduced):
                self.B[k, l] = rbt.B[self.controlled_inds[k],
                                     self.controlled_inds[l]]
        if np.linalg.matrix_rank(self.B) < self.nq_reduced:
            print "The joint set specified is underactuated."
            sys.exit(-1)
        self.B_inv = np.linalg.inv(self.B)
        # Copy lots of stuff
        self.rbt = rbt
        self.nq = rbt.get_num_positions()
        self.plant = plant
        self.nu = plant.get_input_port(0).size()
        self.print_period = print_period
        self.last_print_time = -print_period
        self.shut_up = False

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())
        self.plan_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued, 0)

        self._DeclareDiscreteState(self.nu)
        self._DeclarePeriodicDiscreteUpdate(period_sec=control_period)
        self._DeclareVectorOutputPort(
            BasicVector(self.nu),
            self._DoCalcVectorOutput)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        # (This makes sure relevant event handlers get called.)
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        new_control_input = discrete_state. \
            get_mutable_vector().get_mutable_value()
        t= context.get_time()
        x = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()
        plan = self.EvalAbstractInput(
            context, self.plan_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]
        kinsol = self.rbt.doKinematics(q, v)
        qdd_desired = np.zeros(self.rbt.get_num_positions())

        if plan.type == "JointSpacePlan":
            q_ref = plan.traj.value(t - plan.t_start).flatten()
            v_ref = plan.traj_d.value(t - plan.t_start).flatten()

            qerr = (q_ref[self.controlled_inds] - q[self.controlled_inds])
            verr = (v_ref[self.controlled_inds] - v[self.controlled_inds])

            # Get the full LHS of the manipulator equations
            # given the current config and desired accelerations
            qdd_desired[self.controlled_inds] = 1000.*qerr + 100*verr
        elif plan.type == "TaskSpacePlan":
            t2 = t - plan.t_start
            H_WE = self.rbt.CalcBodyPoseInWorldFrame(kinsol, self.rbt.get_body(idx_ee))
            H_EW = np.linalg.inv(H_WE)
            R_EW = H_EW[0:3, 0:3]
            p_EW = H_EW[0:3, 3]

            Ad_EW = np.zeros((6, 6))
            Ad_EW[0:3, 0:3] = R_EW
            Ad_EW[3:6, 3:6] = R_EW
            p_tilt = np.array([[0, -p_EW[2], p_EW[1]], \
                               [p_EW[2], 0, -p_EW[1]], \
                               [-p_EW[1], p_EW[0], 0]])
            Ad_EW[3:6, 0:3] = p_tilt.dot(R_EW)

            H_WEr = np.eye(4)
            H_WEr[0:3, 0:3] = plan.R_WE_ref
            H_WEr[0:3, 3] = plan.traj.value(t2).flatten()

            H_EEr = H_EW.dot(H_WEr)
            p_EEr = H_EEr[0:3, 3]
            R_EEr = H_EEr[0:3, 0:3]

            phi = np.arccos(0.5 * (R_EEr[0, 0] + R_EEr[1, 1] + R_EEr[2, 2] - 1))
            phi_over_sin_phi = 0
            if np.abs(phi) < 1e-6:
                phi_over_sin_phi = 1
            else:
                phi_over_sin_phi = phi / np.sin(phi)

            log_R_EEr_matirx = 0.5 * phi_over_sin_phi * (R_EEr - R_EEr.T)
            log_R_EEr = np.zeros(3)
            log_R_EEr[0] = log_R_EEr_matirx[2, 1]
            log_R_EEr[1] = log_R_EEr_matirx[0, 2]
            log_R_EEr[2] = log_R_EEr_matirx[1, 0]

            T_WE_E = self.rbt.relativeTwist(kinsol, idx_world, idx_ee, idx_ee)
            T_WEr_E = np.zeros(6)
            T_WEr_E[3:6] = plan.traj_d.value(t2).flatten()
            T_EEr_E = T_WEr_E - T_WE_E

            kp_rotation = np.full(3, 50)
            kp_translation = np.full(3, 50)
            kd = np.full(6, 5)

            dT_WE_E_des = kd * T_EEr_E
            dT_WE_E_des[0:3] += kp_rotation * log_R_EEr
            dT_WE_E_des[3:6] += kp_translation * p_EEr

            J_WE_E, _ = self.rbt.geometricJacobian(kinsol, idx_world, idx_ee, idx_ee)

            prog = MathematicalProgram()
            kuka_qdd_des = prog.NewContinuousVariables(7, 'kuka_qdd_des')
            lhs = dT_WE_E_des - self.rbt.geometricJacobianDotTimesV(kinsol, idx_world, idx_ee, idx_ee)
            rhs = J_WE_E.dot(kuka_qdd_des)
            # for i in range(lhs.size):
            #     prog.AddLinearConstraint(rhs[i], lhs[i], lhs[i])
            prog.AddQuadraticCost(100 * ((lhs - rhs) ** 2).sum())
            prog.AddQuadraticCost((kuka_qdd_des ** 2).sum())
            prog.Solve()
            kuka_qdd_des_values = prog.GetSolution(kuka_qdd_des)

            qdd_desired[self.controlled_inds] = kuka_qdd_des_values

        lhs = self.rbt.inverseDynamics(kinsol, external_wrenches={}, vd=qdd_desired)
        new_u = self.B_inv.dot(lhs[self.controlled_inds])
        new_control_input[:] = new_u

    def _DoCalcVectorOutput(self, context, y_data):
        if (self.print_period and
                context.get_time() - self.last_print_time
                >= self.print_period):
            print "t: ", context.get_time()
            self.last_print_time = context.get_time()
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = control_output[:]

class HandController(LeafSystem):
    def __init__(self, rbt, plant,
                 control_period=0.001):
        LeafSystem.__init__(self)
        self.set_name("Hand Controller")

        self.controlled_joint_names = [
            "left_finger_sliding_joint",
            "right_finger_sliding_joint"
        ]

        self.max_force = 100.  # gripper max closing / opening force

        self.controlled_inds, _ = kuka_utils.extract_position_indices(
            rbt, self.controlled_joint_names)

        self.nu = plant.get_input_port(1).size()
        self.nq = rbt.get_num_positions()

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())

        self.setpoint_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   1)

        self._DeclareDiscreteState(self.nu)
        self._DeclarePeriodicDiscreteUpdate(period_sec=control_period)
        self._DeclareVectorOutputPort(
            BasicVector(self.nu),
            self._DoCalcVectorOutput)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        # (This makes sure relevant event handlers get called.)
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        new_control_input = discrete_state. \
            get_mutable_vector().get_mutable_value()
        x = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()

        gripper_width_des = self.EvalVectorInput(
            context, self.setpoint_input_port.get_index()).get_value()

        q_full = x[:self.nq]
        v_full = x[self.nq:]

        q = q_full[self.controlled_inds]
        q_des = np.array([-gripper_width_des[0], gripper_width_des[0]])
        v = v_full[self.controlled_inds]
        v_des = np.zeros(2)

        qerr = q_des - q
        verr = v_des - v

        Kp = 1000.
        Kv = 100.
        new_control_input[:] = np.clip(
            Kp * qerr + Kv * verr, -self.max_force, self.max_force)

    def _DoCalcVectorOutput(self, context, y_data):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = control_output[:]


class ManipStateMachine(LeafSystem):
    ''' Encodes the high-level logic
        for the manipulation system.

        This state machine sends 3 plans to the robot controller.
        1. The manipulator starts at qtraj.value(0) and follows
        qtraj (a joint space trajectory) to reach a pre-grasp posture.

        2. The manipulator then moves forward in the world x-axis
        following a straight line in the world Cartesian frame.

        3. Upon contacting the object, the manipulator moves back to
        qtraj.value(0), following a joint space trajectory connecting
        the arm's current posture and qtraj.value(0).
    '''
    def __init__(self, rbt, plant, qtraj, R_WEr):
        LeafSystem.__init__(self)
        self.set_name("Manipulation State Machine")

        self.controlled_inds, _ = kuka_utils.extract_position_indices(
            rbt, kuka_controlled_joint_names)

        self.current_plan = None
        self.needs_new_plan = [True, True, True]
        self.t_xyz_traj = 2.0
        self.t_touch = 0

        self.qtraj = qtraj
        self.R_WEr = R_WEr
        self.rbt = rbt
        self.nq = rbt.get_num_positions()
        self.plant = plant
        self.idx_manipuland = 14
        self.idx_gripper_base = 15

        self.collision_element_to_body_map = kuka_utils.GetCollisionElementToRigidBodyIndexMap(rbt)
        self.is_gripper_base_and_object_in_contact = False

        self.robot_state_input_port = \
            self._DeclareInputPort(PortDataType.kVectorValued,
                                   rbt.get_num_positions() +
                                   rbt.get_num_velocities())
        self.contact_result_input_port = \
            self._DeclareInputPort(PortDataType.kAbstractValued,
                                   plant.contact_results_output_port().size())

        self._DeclareDiscreteState(1)
        self._DeclarePeriodicDiscreteUpdate(period_sec=0.01)
        self.kuka_plan_output_port = \
            self._DeclareAbstractOutputPort(
                lambda: AbstractValue.Make(Plan()), self.CalcPlan)
        self.hand_setpoint_output_port = \
            self._DeclareVectorOutputPort(
                BasicVector(1), self._DoCalcHandSetpointOutput)

        # self._DeclarePeriodicPublish(0.01, 0.0)

    def CalcEEPoseInWorldFrame(self, q):
        kinsol = self.rbt.doKinematics(q, np.zeros(self.nq))
        idx_world = 0
        idx_ee = 13
        H_WE = self.rbt.CalcBodyPoseInWorldFrame(kinsol, self.rbt.get_body(idx_ee))
        return H_WE

    def DetectContactBetweenGripperBaseAndObject(self, contact_results, t):
        for contact_i in range(contact_results.get_num_contacts()):
            contact_info = contact_results.get_contact_info(contact_i)
            contact_force = contact_info.get_resultant_force()
            is_contact_1 = \
                self.collision_element_to_body_map[contact_info.get_element_id_1()] == self.idx_gripper_base \
                and \
                self.collision_element_to_body_map[contact_info.get_element_id_2()] == self.idx_manipuland
            is_contact_2 = \
                self.collision_element_to_body_map[contact_info.get_element_id_2()] == self.idx_gripper_base \
                and \
                self.collision_element_to_body_map[contact_info.get_element_id_1()] == self.idx_manipuland
            if is_contact_1 or is_contact_2:
                self.is_gripper_base_and_object_in_contact = True
                self.t_touch = t
                print("gripper base and object are in contact.")
                break


    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(
            self, context, events, discrete_state)

        new_state = discrete_state. \
            get_mutable_vector().get_mutable_value()
        # Close gripper after plan has been executed
        if self.is_gripper_base_and_object_in_contact:
            new_state[:] = 0.
        else:
            new_state[:] = 0.1

    def CalcPlan(self, context, y_data):
        t = context.get_time()
        x = self.EvalVectorInput(
            context, self.robot_state_input_port.get_index()).get_value()
        q = x[:self.nq]
        v = x[self.nq:]

        # handle contact
        if not self.is_gripper_base_and_object_in_contact:
            contact_results = self.EvalAbstractInput(context, 1).get_value()
            self.DetectContactBetweenGripperBaseAndObject(contact_results, t)

        if t < self.qtraj.end_time() + 1.0:
            if self.needs_new_plan[0]:
                self.needs_new_plan[0] = False
                self.current_plan = Plan(type = plan_types[0],
                                         trajectory = self.qtraj,
                                         start_time = 0)
            y_data.set_value(self.current_plan)
        else:
            if not self.is_gripper_base_and_object_in_contact:
                if self.needs_new_plan[1]:
                    self.needs_new_plan[1] = False

                    kinsol = self.rbt.doKinematics(q, v)
                    H_WE = self.rbt.CalcBodyPoseInWorldFrame(kinsol, self.rbt.get_body(idx_ee))
                    p_WE = H_WE[0:3, 3]

                    p_WEr_start = p_WE
                    p_WEr_end = p_WE + np.array([0.3, 0.0, 0.0])
                    times = np.array([0, self.t_xyz_traj])
                    self.xyz_traj = PiecewisePolynomial.FirstOrderHold(
                        times, np.vstack((p_WEr_start, p_WEr_end)).T)
                    self.current_plan = Plan(type = plan_types[1],
                                             trajectory = self.xyz_traj,
                                             start_time = self.qtraj.end_time() + 1.0,
                                             R_WE_ref = self.R_WEr)

                y_data.set_value(self.current_plan)
            else:
                if self.needs_new_plan[2]:
                    self.needs_new_plan[2] = False

                    times = [0, 0.5, 0.5 + self.qtraj.end_time()]
                    knots = np.vstack((q, q, self.qtraj.value(0).flatten()))
                    self.current_plan = Plan(
                        type=plan_types[0],
                        trajectory=PiecewisePolynomial.FirstOrderHold(times, knots.T),
                        start_time=self.t_touch)

                y_data.set_value(self.current_plan)

    def _DoCalcHandSetpointOutput(self, context, y_data):
        state = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        # Get the ith finger control output
        y[:] = state[0]