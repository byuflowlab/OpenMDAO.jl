import juliacall; jl = juliacall.newmodule("OpenMDAOJuliaComps")
from types import MethodType

import numpy as np

import openmdao.api as om
from openmdao.core.analysis_error import AnalysisError

from juliacall import JuliaError
# This imports the Julia package OpenMDAOCore:
jl.seval("using OpenMDAOCore: OpenMDAOCore")


def _initialize_common(self):
    self.options.declare('jlcomp', recordable=False)
    self.options.declare('noisy_julia_domain_error', default=False, types=bool)


def _setup_common(self):
    self._jlcomp = self.options['jlcomp']
    input_data, output_data, partials_data = jl.OpenMDAOCore.setup(self._jlcomp)

    for var in input_data:
        if var.tags is not None:
            tags = list(var.tags)
        else:
            tags = None
        if var.shape_by_conn or var.copy_shape:
            shape = None
        else:
            shape = var.shape
        self.add_input(var.name, shape=shape, val=var.val,
                       units=var.units, tags=tags, shape_by_conn=var.shape_by_conn,
                       copy_shape=var.copy_shape)

    for var in output_data:
        if var.tags is not None:
            tags = list(var.tags)
        else:
            tags = None
        if var.shape_by_conn or var.copy_shape:
            shape = None
        else:
            shape = var.shape
        self.add_output(var.name, shape=shape, val=var.val,
                        units=var.units, lower=var.lower, upper=var.upper, tags=tags,
                        shape_by_conn=var.shape_by_conn,
                        copy_shape=var.copy_shape)

    for data in partials_data:
        self.declare_partials(data.of, data.wrt,
                              rows=data.rows, cols=data.cols,
                              val=data.val, method=data.method)


def _setup_partials_common(self):
    if jl.OpenMDAOCore.has_setup_partials(self._jlcomp):
        # Ignore the partials data from `setup`, since we've already passed that to `declare_partials` in `_setup_common`.
        input_data, output_data, _ = jl.OpenMDAOCore.setup(self._jlcomp)

        # Build up a dict mapping the input names to their size.
        input_sizes_d = {}
        for vd in input_data:
            if vd.shape_by_conn or vd.copy_shape:
                input_sizes_d[vd.name] = self._get_var_meta(vd.name, "shape")
        input_sizes = juliacall.convert(jl.Dict, input_sizes_d)

        # Build up a dict mapping the output names to their size.
        output_sizes_d = {}
        for vd in output_data:
            if vd.shape_by_conn or vd.copy_shape:
                output_sizes_d[vd.name] = self._get_var_meta(vd.name, "shape")
        output_sizes = juliacall.convert(jl.Dict, output_sizes_d)

        jlcomp_new, partials_data = jl.OpenMDAOCore.setup_partials(self._jlcomp, input_sizes, output_sizes)
        self._jlcomp = self.options["jlcomp"] = jlcomp_new

        for data in partials_data:
            self.declare_partials(data.of, data.wrt,
                                  rows=data.rows, cols=data.cols,
                                  val=data.val, method=data.method)


def _only(x):
    [val] = x
    return val


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
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})

                partials_dict = {}
                for (of_abs, wrt_abs), subjac in partials.items():
                    of_rel = of_abs.split(".")[-1]
                    wrt_rel = wrt_abs.split(".")[-1]
                    partials_dict[of_rel, wrt_rel] = np.atleast_1d(subjac)
                partials_dict = juliacall.convert(jl.Dict, partials_dict)

                try:
                    jl.OpenMDAOCore.compute_partials_b(self._jlcomp, inputs_dict, partials_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.compute_partials:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.compute_partials:\n{e}")
                    else:
                        raise e from None

                # Handle scalar entries in partials, which aren't passed by reference when constructing partials_dict.
                for (of_abs, wrt_abs) in list(partials.keys()):
                    subjac = partials[of_abs, wrt_abs]
                    if not isinstance(subjac, np.ndarray):
                        of_rel = of_abs.split(".")[-1]
                        wrt_rel = wrt_abs.split(".")[-1]
                        partials[of_obs, wrt_abs] = _only(partials_dict[of_rel, wrt_rel])

            # https://www.ianlewis.org/en/dynamically-adding-method-classes-or-class-instanc
            self.compute_partials = MethodType(compute_partials, self)
            # Hmm...
            self._has_compute_partials = True

        if jl.OpenMDAOCore.has_compute_jacvec_product(self._jlcomp):
            def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
                d_inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_inputs.items()})
                d_outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_outputs.items()})

                try:
                    jl.OpenMDAOCore.compute_jacvec_product_b(self._jlcomp, inputs_dict, d_inputs_dict, d_outputs_dict, mode)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.compute_jacvec_product:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.compute_jacvec_product:\n{e}")
                    else:
                        raise e from None

                if mode == "fwd":
                    # Handle scalar entries in d_outputs, which aren't passed by reference when constructing doutputs_dict.
                    for k in list(d_outputs.keys()):
                        if not isinstance(d_outputs[k], np.ndarray):
                            d_outputs[k] = _only(d_outputs_dict[k])
                elif mode == "rev":
                    # Handle scalar entries in d_inputs, which aren't passed by reference when constructing dinputs_dict.
                    for k in list(d_inputs.keys()):
                        if not isinstance(d_inputs[k], np.ndarray):
                            d_inputs[k] = _only(d_inputs_dict[k])
                else:
                    raise ValueError(f"unknown mode = {mode} in {self}.compute_jacvec_product")


            self.compute_jacvec_product = MethodType(compute_jacvec_product, self)
            # https://github.com/OpenMDAO/OpenMDAO/pull/2802
            self.matrix_free = True

    def setup_partials(self):
        _setup_partials_common(self)


    def compute(self, inputs, outputs):
        inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
        outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in outputs.items()})

        try:
            jl.OpenMDAOCore.compute_b(self._jlcomp, inputs_dict, outputs_dict)
        except JuliaError as e:
            if jl.isa(e.exception, jl.DomainError):
                if self.options['noisy_julia_domain_error']:
                    print(f"caught Julia DomainError in {self}.compute:\n{e}")
                raise AnalysisError(f"caught Julia DomainError in {self}.compute:\n{e}")
            else:
                raise e from None

        # Handle scalar entries in outputs, which aren't passed by reference when constructing outputs_dict.
        for k in list(outputs_dict.keys()):
            if not isinstance(outputs[k], np.ndarray):
                outputs[k] = _only(outputs_dict[k])


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
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
                outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in outputs.items()})
                residuals_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in residuals.items()})

                try:
                    jl.OpenMDAOCore.apply_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.apply_nonlinear:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.apply_nonlinear:\n{e}")
                    else:
                        raise e from None

                # Handle scalar entries in residuals, which aren't passed by reference when constructing residuals_dict.
                for k in list(residuals.keys()):
                    if not isinstance(residuals[k], np.ndarray):
                        residuals[k] = _only(residuals_dict[k])

            self.apply_nonlinear = MethodType(apply_nonlinear, self)

        if jl.OpenMDAOCore.has_solve_nonlinear(self._jlcomp):
            def solve_nonlinear(self, inputs, outputs):
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
                outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in outputs.items()})

                try:
                    jl.OpenMDAOCore.solve_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.solve_nonlinear:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.solve_nonlinear:\n{e}")
                    else:
                        raise e from None

                # Handle scalar entries in outputs, which aren't passed by reference when constructing outputs_dict.
                for k in list(outputs.keys()):
                    if not isinstance(outputs[k], np.ndarray):
                        outputs[k] = _only(outputs_dict[k])

            self.solve_nonlinear = MethodType(solve_nonlinear, self)
            # https://github.com/OpenMDAO/OpenMDAO/pull/2802
            self._has_solve_nl = True

        if jl.OpenMDAOCore.has_linearize(self._jlcomp):
            def linearize(self, inputs, outputs, partials):
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
                outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in outputs.items()})

                partials_dict = {}
                for (of_abs, wrt_abs), subjac in partials.items():
                    of_rel = of_abs.split(".")[-1]
                    wrt_rel = wrt_abs.split(".")[-1]
                    partials_dict[of_rel, wrt_rel] = np.atleast_1d(subjac)
                partials_dict = juliacall.convert(jl.Dict, partials_dict)

                try:
                    jl.OpenMDAOCore.linearize_b(self._jlcomp, inputs_dict, outputs_dict, partials_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.linearize:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.linearize:\n{e}")
                    else:
                        raise e from None

                # Handle scalar entries in partials, which aren't passed by reference when constructing partials_dict.
                for (of_abs, wrt_abs) in list(partials.keys()):
                    subjac = partials[of_abs, wrt_abs]
                    if not isinstance(subjac, np.ndarray):
                        of_rel = of_abs.split(".")[-1]
                        wrt_rel = wrt_abs.split(".")[-1]
                        partials[of_obs, wrt_abs] = _only(partials_dict[of_rel, wrt_rel])

            self.linearize = MethodType(linearize, self)
            self._has_linearize = True

        if jl.OpenMDAOCore.has_apply_linear(self._jlcomp):
            def apply_linear(self, inputs, outputs, d_inputs, d_outputs, d_residuals, mode):
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
                outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in outputs.items()})
                d_inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_inputs.items()})
                d_outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_outputs.items()})
                d_residuals_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_residuals.items()})

                try:
                    jl.OpenMDAOCore.apply_linear_b(self._jlcomp, inputs_dict, outputs_dict,
                            d_inputs_dict, d_outputs_dict, d_residuals_dict, mode)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.apply_linear:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.apply_linear:\n{e}")
                    else:
                        raise e from None

                if mode == "fwd":
                    # Handle scalar entries in d_residuals, which aren't passed by reference when constructing d_residuals_dict.
                    for k in list(d_residuals.keys()):
                        if not isinstance(d_residuals[k], np.ndarray):
                            d_residuals[k] = _only(d_residuals_dict[k])
                elif mode == "rev":
                    # Handle scalar entries in d_inputs, which aren't passed by reference when constructing d_inputs_dict.
                    for k in list(d_inputs.keys()):
                        if not isinstance(d_inputs[k], np.ndarray):
                            d_inputs[k] = _only(d_inputs_dict[k])
                    # Handle scalar entries in d_outputs, which aren't passed by reference when constructing d_outputs_dict.
                    for k in list(d_outputs.keys()):
                        if not isinstance(d_outputs[k], np.ndarray):
                            d_outputs[k] = _only(d_outputs_dict[k])
                else:
                    raise ValueError(f"unknown mode = {mode} in {self}.apply_linear")

            self.apply_linear = MethodType(apply_linear, self)
            # https://github.com/OpenMDAO/OpenMDAO/pull/2802
            self.matrix_free = True

        if jl.OpenMDAOCore.has_solve_linear(self._jlcomp):
            def solve_linear(self, d_outputs, d_residuals, mode):
                d_outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_outputs.items()})
                d_residuals_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in d_residuals.items()})

                try:
                    jl.OpenMDAOCore.solve_linear_b(self._jlcomp, d_outputs_dict, d_residuals_dict, mode)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.solve_linear:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.solve_linear:\n{e}")
                    else:
                        raise e from None

                if mode == "fwd":
                    # Handle scalar entries in d_outputs, which aren't passed by reference when constructing d_outputs_dict.
                    for k in list(d_outputs.keys()):
                        if not isinstance(d_outputs[k], np.ndarray):
                            d_outputs[k] = _only(d_outputs_dict[k])
                elif mode == "rev":
                    # Handle scalar entries in d_residuals, which aren't passed by reference when constructing d_residuals_dict.
                    for k in list(d_residuals.keys()):
                        if not isinstance(d_residuals[k], np.ndarray):
                            d_residuals[k] = _only(d_residuals_dict[k])
                else:
                    raise ValueError(f"unknown mode = {mode} in {self}.solve_linear")

            self.solve_linear = MethodType(solve_linear, self)

    def setup_partials(self):
        _setup_partials_common(self)

    def _configure(self):
        super()._configure()

        if jl.OpenMDAOCore.has_guess_nonlinear(self._jlcomp):
            def guess_nonlinear(self, inputs, outputs, residuals):
                inputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in inputs.items()})
                outputs_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in outputs.items()})
                residuals_dict = juliacall.convert(jl.Dict, {k: np.atleast_1d(v) for k, v in residuals.items()})

                try:
                    jl.OpenMDAOCore.guess_nonlinear_b(self._jlcomp, inputs_dict, outputs_dict, residuals_dict)
                except JuliaError as e:
                    if jl.isa(e.exception, jl.DomainError):
                        if self.options['noisy_julia_domain_error']:
                            print(f"caught Julia DomainError in {self}.guess_nonlinear:\n{e}")
                        raise AnalysisError(f"caught Julia DomainError in {self}.guess_nonlinear:\n{e}")
                    else:
                        raise e from None

                # Handle scalar entries in outputs, which aren't passed by reference when constructing outputs_dict.
                for k in list(outputs.keys()):
                    if not isinstance(outputs[k], np.ndarray):
                        outputs[k] = _only(outputs_dict[k])

            self.guess_nonlinear = MethodType(guess_nonlinear, self)
            # Hmm...
            self._has_guess = True


def to_jlsymstrdict(d):
    T = juliacall.Main.Dict[juliacall.Main.Symbol,juliacall.Main.String]
    return juliacall.convert(T, d)

def to_jlsymdictstranydict(d):
    T = juliacall.Main.Dict[
            juliacall.Main.Symbol,
            juliacall.Main.Dict[
                juliacall.Main.String, juliacall.Main.Any]]
    return juliacall.convert(T, d)
