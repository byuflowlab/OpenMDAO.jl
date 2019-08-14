import openmdao.api as om


def _julia_initialize(self):
    self.options.declare('julia_comp_data')


def _julia_setup(self):
    comp_data = self.options['julia_comp_data']
    input_data = comp_data.inputs
    output_data = comp_data.outputs

    for var in input_data:
        self.add_input(var.name, shape=tuple(var.shape), val=var.val,
                       units=var.units)

    for var in output_data:
        self.add_output(var.name, shape=tuple(var.shape), val=var.val,
                        units=var.units)

    for data in comp_data.partials:
        self.declare_partials(data.of, data.wrt,
                              rows=data.rows, cols=data.cols,
                              val=data.val)


class JuliaExplicitComp(om.ExplicitComponent):

    def initialize(self):
        _julia_initialize(self)

    def setup(self):
        _julia_setup(self)

    def compute(self, inputs, outputs):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)

        comp_data.compute(comp_data.self, inputs_dict, outputs_dict)

    def compute_partials(self, inputs, partials):
        comp_data = self.options['julia_comp_data']
        if comp_data.compute_partials:
            inputs_dict = dict(inputs)

            partials_dict = {}
            for part_names in comp_data.partials:
                of_wrt = part_names.of, part_names.wrt
                partials_dict[of_wrt] = partials[of_wrt]

            comp_data.compute_partials(comp_data.self, inputs_dict,
                                       partials_dict)


class JuliaImplicitComp(om.ImplicitComponent):

    def initialize(self):
        _julia_initialize(self)

    def setup(self):
        _julia_setup(self)

    def apply_nonlinear(self, inputs, outputs, residuals):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)
        residuals_dict = dict(residuals)

        comp_data.apply_nonlinear(comp_data.self, inputs_dict, outputs_dict,
                                  residuals_dict)

    def linearize(self, inputs, outputs, partials):
        comp_data = self.options['julia_comp_data']
        if comp_data.linearize:
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)

            partials_dict = {}
            for part_names in comp_data.partials:
                of_wrt = part_names.of, part_names.wrt
                partials_dict[of_wrt] = partials[of_wrt]

            comp_data.linearize(comp_data.self, inputs_dict, outputs_dict,
                                partials_dict)

    def guess_nonlinear(self, inputs, outputs, residuals):
        comp_data = self.options['julia_comp_data']
        if comp_data.guess_nonlinear:
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)
            residuals_dict = dict(residuals)

            comp_data.guess_nonlinear(comp_data.self, inputs_dict,
                                      outputs_dict, residuals_dict)
