import numpy as np

from pydrake.all import (
    LeafSystem,
    PortDataType,
    DiagramBuilder,
    SignalLogger,
    Simulator,
    AbstractValue,
    BasicVector
)

class DummyPlan:
    def __init__(self, type, index):
        self.type = type
        self.index = index


class AbstractSender(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self._DeclareAbstractOutputPort(lambda: AbstractValue.Make(DummyPlan(0, 0)), self.CalcPlan)
        self.plans = [DummyPlan("JointSpacePlan", 0),
                      DummyPlan("TaskSpacePlan", 1) ]

    def CalcPlan(self, context, y_data):
        t = context.get_time()
        if t <= 1:
            y_data.set_value(self.plans[0])
        else:
            y_data.set_value(self.plans[1])


class AbstractReceiver(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self._DeclareInputPort(PortDataType.kAbstractValued, 0)
        self._DeclareVectorOutputPort(BasicVector(1), self._DoCalcVectorOutput)

        self._DeclareDiscreteState(1)
        self._DeclarePeriodicDiscreteUpdate(period_sec=0.1)

    def _DoCalcDiscreteVariableUpdates(self, context, events, discrete_state):
        # Call base method to ensure we do not get recursion.
        LeafSystem._DoCalcDiscreteVariableUpdates(self, context, events, discrete_state)

        plan = self.EvalAbstractInput(context, 0).get_value()
        new_index = discrete_state.get_mutable_vector().get_mutable_value()
        new_index[0] = plan.index

    def _DoCalcVectorOutput(self, context, y_data):
        control_output = context.get_discrete_state_vector().get_value()
        y = y_data.get_mutable_value()
        y[:] = control_output


if __name__ == "__main__":
    builder = DiagramBuilder()
    sender = builder.AddSystem(AbstractSender())
    receiver = builder.AddSystem(AbstractReceiver())
    logger = builder.AddSystem(SignalLogger(1))
    # logger._DeclarePeriodicPublish(0.1)
    builder.Connect(sender.get_output_port(0), receiver.get_input_port(0))
    builder.Connect(receiver.get_output_port(0), logger.get_input_port(0))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()
    simulator.set_target_realtime_rate(0)
    simulator.StepTo(2)











