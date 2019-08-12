import openmdao.api as om


def _julia_init(self, **kwargs):

    comp_data = self.options['julia_comp_data']
    self._julia_options = {}
    for option in comp_data.options:
        self.options.declare(option.name,
                             # types=option.type,
                             default=option.val)
        self._julia_options[option.name] = option.val

    self.options.update(kwargs)


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

    def __init__(self, **kwargs):
        super(JuliaExplicitComp, self).__init__(**kwargs)
        _julia_init(self, **kwargs)

    def initialize(self):
        _julia_initialize(self)

    def setup(self):
        _julia_setup(self)

    def compute(self, inputs, outputs):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)

        comp_data.compute(self._julia_options, inputs_dict, outputs_dict)

    def compute_partials(self, inputs, partials):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)

        partials_dict = {}
        for part_names in comp_data.partials:
            of_wrt = part_names.of, part_names.wrt
            partials_dict[of_wrt] = partials[of_wrt]

        comp_data.compute_partials(self._julia_options, inputs_dict,
                                   partials_dict)


class JuliaImplicitComp(om.ImplicitComponent):

    def __init__(self, **kwargs):
        super(JuliaImplicitComp, self).__init__(**kwargs)
        _julia_init(self, **kwargs)

    def initialize(self):
        _julia_initialize(self)

    def setup(self):
        _julia_setup(self)

    def apply_nonlinear(self, inputs, outputs, residuals):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)
        residuals_dict = dict(residuals)

        comp_data.apply_nonlinear(self._julia_options, inputs_dict,
                                  outputs_dict, residuals_dict)

    def linearize(self, inputs, outputs, partials):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)

        partials_dict = {}
        for part_names in comp_data.partials:
            of_wrt = part_names.of, part_names.wrt
            partials_dict[of_wrt] = partials[of_wrt]

        comp_data.linearize(self._julia_options, inputs_dict, outputs_dict,
                            partials_dict)
        # print(f"partials_dict['Np', 'phi'] = {partials_dict['Np', 'phi']}")
        # print(f"partials['Np', 'phi'] = {partials['Np', 'phi']}")

    def guess_nonlinear(self, inputs, outputs, residuals):
        comp_data = self.options['julia_comp_data']
        if comp_data.guess_nonlinear:
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)
            residuals_dict = dict(residuals)

            comp_data.guess_nonlinear(self._julia_options, inputs_dict,
                                      outputs_dict, residuals_dict)
