import time

import openmdao.api as om

import omjl




def _initialize_common(self):
    self.options.declare('jl_id')


def _setup_common(self):
    jl_id = self.jl_id = self.options['jl_id']
    self._julia_setup = omjl.setup(jl_id)
    input_data, output_data, partials_data = self._julia_setup(jl_id)

    for var in input_data:
        self.add_input(var.name, shape=var.shape, val=var.val,
                       units=var.units)

    for var in output_data:
        self.add_output(var.name, shape=var.shape, val=var.val,
                        units=var.units)

    for data in partials_data:
        self.declare_partials(data.of, data.wrt,
                              rows=data.rows, cols=data.cols,
                              val=data.val)


class JuliaExplicitComp(om.ExplicitComponent):

    def initialize(self):
        _initialize_common(self)

    def setup(self):
        _setup_common(self)
        self._julia_compute = omjl.compute(self.jl_id)
        self._julia_compute_partials = omjl.compute_partials(self.jl_id)

    def compute(self, inputs, outputs):
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)

        self._julia_compute(self.jl_id, inputs_dict, outputs_dict)

    def compute_partials(self, inputs, partials):
        if self._julia_compute_partials:
            inputs_dict = dict(inputs)

            partials_dict = {}
            for of_wrt in self._declared_partials:
                partials_dict[of_wrt] = partials[of_wrt]

            self._julia_compute_partials(self.jl_id, inputs_dict, partials_dict)


class JuliaImplicitComp(om.ImplicitComponent):

    def initialize(self):
        _initialize_common(self)

    def setup(self):
        _setup_common(self)
        self._julia_apply_nonlinear = omjl.apply_nonlinear(self.jl_id)
        self._julia_linearize = omjl.linearize(self.jl_id)
        self._julia_guess_nonlinear = omjl.guess_nonlinear(self.jl_id)
        self._julia_solve_nonlinear = omjl.solve_nonlinear(self.jl_id)

    def apply_nonlinear(self, inputs, outputs, residuals):
        inputs_dict = dict(inputs)
        outputs_dict = dict(outputs)
        residuals_dict = dict(residuals)

        self._julia_apply_nonlinear(self.jl_id, inputs_dict, outputs_dict,
                                    residuals_dict)

    def linearize(self, inputs, outputs, partials):
        if self._julia_linearize:
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)

            partials_dict = {}
            for of_wrt in self._declared_partials:
                partials_dict[of_wrt] = partials[of_wrt]

            self._julia_linearize(self.jl_id, inputs_dict, outputs_dict,
                                  partials_dict)

    def guess_nonlinear(self, inputs, outputs, residuals):
        if self._julia_guess_nonlinear:
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)
            residuals_dict = dict(residuals)

            self._julia_guess_nonlinear(self.jl_id, inputs_dict, outputs_dict,
                                        residuals_dict)

    def solve_nonlinear(self, inputs, outputs):
        if self._julia_solve_nonlinear:
            inputs_dict = dict(inputs)
            outputs_dict = dict(outputs)

            self._julia_solve_nonlinear(self.jl_id, inputs_dict, outputs_dict)
