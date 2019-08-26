import openmdao.api as om
from julia.OpenMDAO import (get_pycompute, get_pycompute_partials,
                            get_pyapply_nonlinear, get_pylinearize,
                            get_pyguess_nonlinear, get_pysolve_nonlinear)


def _julia_initialize(self):
    self.options.declare('julia_comp_data')


def _julia_setup(self):
    comp_data = self.options['julia_comp_data']
    input_data = comp_data.inputs
    output_data = comp_data.outputs
    partials_data = comp_data.partials

    for var in input_data:
        self.add_input(var.name, shape=tuple(var.shape), val=var.val,
                       units=var.units)

    for var in output_data:
        self.add_output(var.name, shape=tuple(var.shape), val=var.val,
                        units=var.units)

    for data in partials_data:
        self.declare_partials(data.of, data.wrt,
                              rows=data.rows, cols=data.cols,
                              val=data.val)


class JuliaExplicitComp(om.ExplicitComponent):

    def initialize(self):
        _julia_initialize(self)

    def setup(self):
        _julia_setup(self)
        comp_data = self.options['julia_comp_data']
        self._julia_compute = get_pycompute(comp_data)
        self._julia_compute_partials = get_pycompute_partials(comp_data)

    def compute(self, inputs, outputs):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)

        self._julia_compute(comp_data, inputs_dict, outputs_dict)

    def compute_partials(self, inputs, partials):
        comp_data = self.options['julia_comp_data']
        if self._julia_compute_partials:
            inputs_dict = dict(inputs)

            partials_dict = {}
            for part_names in comp_data.partials:
                of_wrt = part_names.of, part_names.wrt
                partials_dict[of_wrt] = partials[of_wrt]

            self._julia_compute_partials(comp_data, inputs_dict, partials_dict)


class JuliaImplicitComp(om.ImplicitComponent):

    def initialize(self):
        _julia_initialize(self)

    def setup(self):
        _julia_setup(self)
        comp_data = self.options['julia_comp_data']
        self._julia_apply_nonlinear = get_pyapply_nonlinear(comp_data)
        self._julia_linearize = get_pylinearize(comp_data)
        self._julia_guess_nonlinear = get_pyguess_nonlinear(comp_data)
        self._julia_solve_nonlinear = get_pysolve_nonlinear(comp_data)

    def apply_nonlinear(self, inputs, outputs, residuals):
        comp_data = self.options['julia_comp_data']
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)
        residuals_dict = dict(residuals)

        self._julia_apply_nonlinear(comp_data, inputs_dict, outputs_dict,
                                    residuals_dict)

    def linearize(self, inputs, outputs, partials):
        if self._julia_linearize:
            comp_data = self.options['julia_comp_data']
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)

            partials_dict = {}
            for part_names in comp_data.partials:
                of_wrt = part_names.of, part_names.wrt
                partials_dict[of_wrt] = partials[of_wrt]

            self._julia_linearize(comp_data, inputs_dict, outputs_dict,
                                  partials_dict)

    def guess_nonlinear(self, inputs, outputs, residuals):
        if self._julia_guess_nonlinear:
            comp_data = self.options['julia_comp_data']
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)
            residuals_dict = dict(residuals)

            self._julia_guess_nonlinear(comp_data, inputs_dict, outputs_dict,
                                        residuals_dict)

    def solve_nonlinear(self, inputs, outputs):
        if self._julia_solve_nonlinear:
            comp_data = self.options['julia_comp_data']
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)

            self._julia_solve_nonlinear(comp_data, inputs_dict, outputs_dict)
