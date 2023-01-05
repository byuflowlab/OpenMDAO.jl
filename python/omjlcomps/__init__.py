from types import MethodType

import openmdao.api as om
from openmdao.core.analysis_error import AnalysisError

import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaComps")
from juliacall import JuliaError
# This imports the Julia package OpenMDAOCore:
jl.seval("using OpenMDAOCore: OpenMDAOCore")


def _initialize_common(self):
    self.options.declare('jlcomp', recordable=False)


def _setup_common(self):
    self._jlcomp = self.options['jlcomp']
    input_data, output_data, partials_data = jl.OpenMDAOCore.setup(self._jlcomp)

    for var in input_data:
        if var.tags is not None:
            tags = list(var.tags)
        else:
            tags = None
        self.add_input(var.name, shape=var.shape, val=var.val,
                       units=var.units, tags=tags)

    for var in output_data:
        if var.tags is not None:
            tags = list(var.tags)
        else:
            tags = None
        self.add_output(var.name, shape=var.shape, val=var.val,
                        units=var.units, lower=var.lower, upper=var.upper, tags=tags)

    for data in partials_data:
        self.declare_partials(data.of, data.wrt,
                              rows=data.rows, cols=data.cols,
                              val=data.val, method=data.method)


class JuliaExplicitComp(om.ExplicitComponent):
    """
    Class for implementing an OpenMDAO.ExplicitComponent using OpenMDAO.jl and the Julia programming language.

    Parameters
    ----------
    jlcomp : subtype of `OpenMDAOCore.AbstractExplicitComp`
        A Julia struct that subtypes `OpenMDAOCore.AbstractExplicitComp`.
        Used by `JuliaExplicitComp` to call Julia functions that mimic methods required by an OpenMDAO `ExplicitComponent` (e.g., `OpenMDAOCore.setup`, `OpenMDAOCore.compute!`, `OpenMDAOCore.compute_partials!`, etc.).
    """

    def initialize(self):
        _initialize_common(self)

    def setup(self):
        _setup_common(self)

        if jl.OpenMDAOCore.has_compute_partials(self._jlcomp):
            def compute_partials(self, inputs, partials):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))

                partials_dict = {}
                for of_wrt in self._declared_partials:
                    partials_dict[of_wrt] = partials[of_wrt]
                partials_dict = juliacall.convert(jl.Dict, partials_dict)

                try:
                    jl.OpenMDAOCore.compute_partials_b(self._jlcomp, inputs_dict, partials_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.compute_partials:\n{e}")
                    else:
                        raise e from None

            # https://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc
            self.compute_partials = MethodType(compute_partials, self)
            # Hmm...
            self._has_compute_partials = True

        if jl.OpenMDAOCore.has_compute_jacvec_product(self._jlcomp):
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                d_inputs_dict = juliacall.convert(jl.Dict, dict(d_inputs))
                d_outputs_dict = juliacall.convert(jl.Dict, dict(d_outputs))

                try:
                    jl.OpenMDAOCore.compute_jacvec_product_b(self._jlcomp, inputs_dict, d_inputs_dict, d_outputs_dict, mode)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.compute_jacvec_product:\n{e}")
                    else:
                        raise e from None

            self.compute_jacvec_product = MethodType(compute_jacvec_product, self)

    def compute(self, inputs, outputs):
        inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
        outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

        try:
            jl.OpenMDAOCore.compute_b(self._jlcomp, inputs_dict, outputs_dict)
        except JuliaError as e:
            if jl.isa(e.exception, jl.DomainError):
                raise AnalysisError(f"caught Julia DomainError in {self}.compute:\n{e}")
            else:
                raise e from None


class JuliaImplicitComp(om.ImplicitComponent):
    """
    Class for implementing an OpenMDAO.ImplicitComponent using OpenMDAO.jl and the Julia programming language.

    Parameters
    ----------
    jlcomp : subtype of `OpenMDAOCore.AbstractImplicitComp`
        A Julia struct that subtypes `OpenMDAOCore.AbstractImplicitComp`.
        Used by `JuliaImplicitComp` to call Julia functions that mimic methods required by an OpenMDAO `ImplicitComponent` (e.g., `OpenMDAOCore.setup`, `OpenMDAOCore.apply_nonlinear!`, `OpenMDAOCore.linearize!`, etc.).
    """

    def initialize(self):
        _initialize_common(self)

    def setup(self):
        _setup_common(self)

        if jl.OpenMDAOCore.has_apply_nonlinear(self._jlcomp):
            def apply_nonlinear(self, inputs, outputs, residuals):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
                residuals_dict = juliacall.convert(jl.Dict, dict(residuals))

                try:
                    jl.OpenMDAOCore.apply_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.apply_nonlinear:\n{e}")
                    else:
                        raise e from None

            self.apply_nonlinear = MethodType(apply_nonlinear, self)

        if jl.OpenMDAOCore.has_solve_nonlinear(self._jlcomp):
            def solve_nonlinear(self, inputs, outputs):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

                try:
                    jl.OpenMDAOCore.solve_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.solve_nonlinear:\n{e}")
                    else:
                        raise e from None

            self.solve_nonlinear = MethodType(solve_nonlinear, self)

        if jl.OpenMDAOCore.has_linearize(self._jlcomp):
            def linearize(self, inputs, outputs, partials):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                outputs_dict = juliacall.convert(jl.Dict, dict(outputs))

                partials_dict = {}
                for of_wrt in self._declared_partials:
                    partials_dict[of_wrt] = partials[of_wrt]
                partials_dict = juliacall.convert(jl.Dict, partials_dict)

                try:
                    jl.OpenMDAOCore.linearize_b(self._jlcomp, inputs_dict, outputs_dict, partials_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.linearize:\n{e}")
                    else:
                        raise e from None

            self.linearize = MethodType(linearize, self)

        if jl.OpenMDAOCore.has_apply_linear(self._jlcomp):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
                d_inputs_dict = juliacall.convert(jl.Dict, dict(d_inputs))
                d_outputs_dict = juliacall.convert(jl.Dict, dict(d_outputs))
                d_residuals_dict = juliacall.convert(jl.Dict, dict(d_residuals))

                try:
                    jl.OpenMDAOCore.apply_linear_b(self._jlcomp, inputs_dict, outputs_dict,
                            d_inputs_dict, d_outputs_dict, d_residuals_dict, mode)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.apply_linear:\n{e}")
                    else:
                        raise e from None

            self.apply_linear = MethodType(apply_linear, self)

        if jl.OpenMDAOCore.has_solve_linear(self._jlcomp):
            def solve_linear(self, d_outputs, d_residuals, mode):
                d_outputs_dict = juliacall.convert(jl.Dict, dict(d_outputs))
                d_residuals_dict = juliacall.convert(jl.Dict, dict(d_residuals))

                try:
                    jl.OpenMDAOCore.solve_linear_b(self._jlcomp, d_outputs_dict, d_residuals_dict, mode)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.solve_linear:\n{e}")
                    else:
                        raise e from None

            # Hello Owen. Red Knights vs Greens.
            self.solve_linear = MethodType(solve_linear, self)


    def _configure(self):
        super()._configure()

        if jl.OpenMDAOCore.has_guess_nonlinear(self._jlcomp):
            def guess_nonlinear(self, inputs, outputs, residuals):
                inputs_dict = juliacall.convert(jl.Dict, dict(inputs))
                outputs_dict = juliacall.convert(jl.Dict, dict(outputs))
                residuals_dict = juliacall.convert(jl.Dict, dict(residuals))

                try:
                    jl.OpenMDAOCore.guess_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        raise AnalysisError(f"caught Julia DomainError in {self}.guess_nonlinear:\n{e}")
                    else:
                        raise e from None

            self.guess_nonlinear = MethodType(guess_nonlinear, self)
            # Hmm...
            self._has_guess = True
