module ICompTest

using OpenMDAOCore
using LinearAlgebra: lu, ldiv!

struct SimpleImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::SimpleImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=2.0),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::SimpleImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.linearize!(self::SimpleImplicit, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

struct SimpleImplicitWithGlob{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::SimpleImplicitWithGlob)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=2.0),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [PartialsData("*", "*"; rows=rows, cols=cols)]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::SimpleImplicitWithGlob, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.linearize!(self::SimpleImplicitWithGlob, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

struct SolveNonlinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::SolveNonlinearImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=2.0),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::SolveNonlinearImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.solve_nonlinear!(self::SolveNonlinearImplicit, inputs, outputs)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y
    @. outputs["z2"] = a*x + y

    return nothing
end

function OpenMDAOCore.linearize!(self::SolveNonlinearImplicit, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

struct MatrixFreeImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::MatrixFreeImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=2.0),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::MatrixFreeImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.apply_linear!(self::MatrixFreeImplicit, inputs, outputs, d_inputs, d_outputs, d_residuals, mode)
    a = self.a
    x, y = inputs["x"], inputs["y"]
    z1, z2 = outputs["z1"], outputs["z2"]

    xdot = get(d_inputs, "x", nothing)
    ydot = get(d_inputs, "y", nothing)
    z1dot = get(d_outputs, "z1", nothing)
    z2dot = get(d_outputs, "z2", nothing)
    Rz1dot = get(d_residuals, "z1", nothing)
    Rz2dot = get(d_residuals, "z2", nothing)

    if mode == "fwd"
        # In forward mode, the goal is to calculate the derivatives of the
        # residuals wrt an upstream input, given the inputs and outputs and the
        # derivatives of the inputs and outputs wrt the upstream input.
        if Rz1dot !== nothing
            fill!(Rz1dot, 0)
            if xdot !== nothing
                @. Rz1dot += 2*a*x*xdot
            end
            if ydot !== nothing
                @. Rz1dot += 2*y*ydot
            end
            if z1dot !== nothing
                @. Rz1dot += -z1dot
            end
        end
        if Rz2dot !== nothing
            fill!(Rz2dot, 0)
            if xdot !== nothing
                @. Rz2dot += a*xdot
            end
            if ydot !== nothing
                @. Rz2dot += ydot
            end
            if z2dot !== nothing
                @. Rz2dot += -z2dot
            end
        end
    elseif mode == "rev"
        # In reverse mode, the goal is to calculate the derivatives of an
        # downstream output wrt the inputs and outputs, given the derivatives of
        # the downstream output wrt the residuals.
        if xdot !== nothing
            fill!(xdot, 0)
            if Rz1dot !== nothing
                @. xdot += 2*a*x*Rz1dot
            end
            if Rz2dot !== nothing
                @. xdot += a*Rz2dot
            end
        end
        if ydot !== nothing
            fill!(ydot, 0)
            if Rz1dot !== nothing
                @. ydot += 2*y*Rz1dot
            end
            if Rz2dot !== nothing
                @. ydot += Rz2dot
            end
        end
        if z1dot !== nothing
            fill!(z1dot, 0)
            if Rz1dot !== nothing
                @. z1dot += -Rz1dot
            end
        end
        if z2dot !== nothing
            fill!(z2dot, 0)
            if Rz2dot !== nothing
                @. z2dot += -Rz2dot
            end
        end
    end
end

struct SolveLinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    a::TF
end

function OpenMDAOCore.setup(self::SolveLinearImplicit)
 
    n = self.n
    inputs = [
        VarData("x"; shape=n, val=2.0),
        VarData("y"; shape=(n,), val=3.0)]

    outputs = [
        VarData("z1"; shape=(n,), val=fill(2.0, n)),
        VarData("z2"; shape=n, val=3.0)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::SolveLinearImplicit, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.solve_nonlinear!(self::SolveLinearImplicit, inputs, outputs)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. outputs["z1"] = a*x*x + y*y
    @. outputs["z2"] = a*x + y

    return nothing
end

function OpenMDAOCore.linearize!(self::SolveLinearImplicit, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

function OpenMDAOCore.solve_linear!(self::SolveLinearImplicit, d_outputs, d_residuals, mode)
    n = self.n
    a = self.a

    z1dot = get(d_outputs, "z1", nothing)
    z2dot = get(d_outputs, "z2", nothing)
    Rz1dot = get(d_residuals, "z1", nothing)
    Rz2dot = get(d_residuals, "z2", nothing)

    if mode == "fwd"
        # In forward mode, the goal is to calculate the total derivatives of the
        # implicit outputs wrt an upstream input, given the
        # derivatives of the residuals wrt the upstream input.
        if z1dot !== nothing
            pRz1_pz1 = zeros(self.n, self.n)
            for i in 1:n
                pRz1_pz1[i, i] = -1
            end
            pRz1_pz1_lu = lu(pRz1_pz1)
            # Annoying: z1dot is a PythonCall.PyArray, which isn't a
            # StridedArray and so can't be used with ldiv! directly.
            z1dotfoo = Vector{eltype(z1dot)}(undef, size(z1dot))
            ldiv!(z1dotfoo, pRz1_pz1_lu, Rz1dot)
            z1dot .= z1dotfoo
        end

        if z2dot !== nothing
            pRz2_pz2 = zeros(self.n, self.n)
            for i in 1:n
                pRz2_pz2[i, i] = -1
            end
            z2dotfoo = Vector{eltype(z2dot)}(undef, size(z2dot))
            # Annoying: z1dot is a PythonCall.PyArray, which isn't a
            # StridedArray and so can't be used with ldiv! directly.
            ldiv!(z2dotfoo, lu(pRz2_pz2), Rz2dot)
            z2dot .= z2dotfoo
        end

    elseif mode == "rev"
        # In reverse mode, the goal is to calculate the total derivatives of a
        # downstream output wrt a residual, given the total derivative of the
        # downstream output wrt the residual.
        if Rz1dot !== nothing
            pRz1_pz1 = zeros(self.n, self.n)
            for i in 1:n
                pRz1_pz1[i, i] = -1
            end
            # The partial derivative of z1's residual wrt z1 is diagonal, so
            # it's equal to it's transpose.
            # ldiv!(z1dot, pRz1_pz1, Rz1dot)
            Rz1dotfoo = Vector{eltype(Rz1dot)}(undef, size(Rz1dot))
            ldiv!(Rz1dotfoo, lu(pRz1_pz1), z1dot)
            Rz1dot .= Rz1dotfoo
        end
        if Rz2dot != nothing
            pRz2_pz2 = zeros(self.n, self.n)
            for i in 1:n
                pRz2_pz2[i, i] = -1
            end
            # The partial derivative of z2's residual wrt z2 is diagonal, so
            # it's equal to it's transpose.
            # ldiv!(z2dot, pRz2_pz2, Rz2dot)
            Rz2dotfoo = Vector{eltype(Rz2dot)}(undef, size(Rz2dot))
            ldiv!(Rz2dotfoo, lu(pRz2_pz2), z2dot)
            Rz2dot .= Rz2dotfoo
        end
    end
    return nothing
end

struct GuessNonlinearImplicit{TI,TF} <: OpenMDAOCore.AbstractImplicitComp
    n::TI  # these would be like "options" in openmdao
    xguess::TF
    xlower::TF
    xupper::TF
end

function OpenMDAOCore.setup(self::GuessNonlinearImplicit)
    n = self.n
    xlower = self.xlower
    xupper = self.xupper
    inputs = [
        VarData("a"; shape=n, val=2.0),
        VarData("b"; shape=(n,), val=3.0),
        VarData("c"; shape=(n,), val=3.0)]

    outputs = [VarData("x"; shape=n, val=3.0, lower=xlower, upper=xupper)]

    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("x", "a"; rows=rows, cols=cols),
        PartialsData("x", "b"; rows, cols),
        PartialsData("x", "c"; rows, cols),
        PartialsData("x", "x"; rows, cols),
    ]

    return inputs, outputs, partials
end

function OpenMDAOCore.apply_nonlinear!(self::GuessNonlinearImplicit, inputs, outputs, residuals)
    a = inputs["a"]
    b = inputs["b"]
    c = inputs["c"]
    x = outputs["x"]
    Rx = residuals["x"]

    @. Rx = a*x^2 + b*x + c

    return nothing
end

function OpenMDAOCore.linearize!(self::GuessNonlinearImplicit, inputs, outputs, partials)
    a = inputs["a"]
    b = inputs["b"]
    c = inputs["c"]
    x = outputs["x"]

    dRx_da = partials["x", "a"]
    dRx_db = partials["x", "b"]
    dRx_dc = partials["x", "c"]
    dRx_dx = partials["x", "x"]

    @. dRx_da = x^2
    @. dRx_db = x
    @. dRx_dc = 1
    @. dRx_dx = 2*a*x + b

    return nothing
end

function OpenMDAOCore.guess_nonlinear!(self::GuessNonlinearImplicit, inputs, outputs, residuals)
    @. outputs["x"] = self.xguess
    return nothing
end

struct ImplicitShapeByConn{TF} <: OpenMDAOCore.AbstractImplicitComp
    a::TF
end

function OpenMDAOCore.setup(self::ImplicitShapeByConn)
    inputs = [
        VarData("x"; val=2.0, shape_by_conn=true),
        VarData("y"; val=3.0, shape_by_conn=true, copy_shape="x")]

    outputs = [
        VarData("z1"; val=2.0, shape_by_conn=true, copy_shape="x"),
        VarData("z2"; val=3.0, shape_by_conn=true, copy_shape="x")]

    partials = []
    return inputs, outputs, partials
end

function OpenMDAOCore.setup_partials(self::ImplicitShapeByConn, input_sizes, output_sizes)
    @assert input_sizes["y"] == input_sizes["x"]
    @assert output_sizes["z1"] == input_sizes["x"]
    @assert output_sizes["z2"] == input_sizes["x"]
    n = only(input_sizes["x"])
    rows = 0:n-1
    cols = 0:n-1
    partials = [
        PartialsData("z1", "x"; rows=rows, cols=cols),
        PartialsData("z1", "y"; rows, cols),
        PartialsData("z1", "z1"; rows, cols),
        PartialsData("z2", "x"; rows, cols),
        PartialsData("z2", "y"; rows, cols),          
        PartialsData("z2", "z2"; rows, cols)
    ]

    return partials
end

function OpenMDAOCore.apply_nonlinear!(self::ImplicitShapeByConn, inputs, outputs, residuals)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. residuals["z1"] = (a*x*x + y*y) - outputs["z1"]
    @. residuals["z2"] = (a*x + y) - outputs["z2"]

    return nothing
end

function OpenMDAOCore.linearize!(self::ImplicitShapeByConn, inputs, outputs, partials)
    a = self.a
    x = inputs["x"]
    y = inputs["y"]

    @. partials["z1", "z1"] = -1.0
    @. partials["z1", "x"] = 2*a*x
    @. partials["z1", "y"] = 2*y

    @. partials["z2", "z2"] = -1.0
    @. partials["z2", "x"] = a
    @. partials["z2", "y"] = 1.0

    return nothing
end

end # module
