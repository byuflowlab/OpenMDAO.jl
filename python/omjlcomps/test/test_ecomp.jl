module ECompTest

using OpenMDAOCore

struct ECompSimple <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::ECompSimple)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompSimple, inputs, outputs)
    outputs["y"][1] = 2*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompSimple, inputs, partials)
    partials["y", "x"][1] = 4*inputs["x"][1]
    return nothing
end

struct ECompSimpleCS <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::ECompSimpleCS)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x"; method="cs")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompSimpleCS, inputs, outputs)
    outputs["y"][1] = 2*inputs["x"][1]^2 + 1
    return nothing
end

struct ECompWithOption <: OpenMDAOCore.AbstractExplicitComp
    a::Float64
end

function OpenMDAOCore.setup(self::ECompWithOption)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompWithOption, inputs, outputs)
    outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompWithOption, inputs, partials)
    partials["y", "x"][1] = 4*self.a*inputs["x"][1]
    return nothing
end

struct ECompWithOptionAndTags <: OpenMDAOCore.AbstractExplicitComp
    a::Float64
    xtag::Vector{String}
    ytag::Vector{String}
end

ECompWithOptionAndTags(a; xtag, ytag) = ECompWithOptionAndTags(a, xtag, ytag)

function OpenMDAOCore.setup(self::ECompWithOptionAndTags)
    input_data = [VarData("x"; tags=self.xtag)]
    output_data = [VarData("y"; tags=self.ytag)]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompWithOptionAndTags, inputs, outputs)
    outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompWithOptionAndTags, inputs, partials)
    partials["y", "x"][1] = 4*self.a*inputs["x"][1]
    return nothing
end

struct ECompWithGlobs <: OpenMDAOCore.AbstractExplicitComp
    a::Float64
    xtag::Vector{String}
    ytag::Vector{String}
end

ECompWithGlobs(a; xtag, ytag) = ECompWithGlobs(a, xtag, ytag)

function OpenMDAOCore.setup(self::ECompWithGlobs)
    input_data = [VarData("x"; tags=self.xtag)]
    output_data = [VarData("y"; tags=self.ytag)]
    partials_data = [PartialsData("*", "*")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompWithGlobs, inputs, outputs)
    outputs["y"][1] = 2*self.a*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompWithGlobs, inputs, partials)
    partials["y", "x"][1] = 4*self.a*inputs["x"][1]
    return nothing
end

struct ECompWithLargeOption <: OpenMDAOCore.AbstractExplicitComp
    a::Vector{Float64}
end

ECompWithLargeOption(n::Integer) = ECompWithLargeOption(range(1.0, n; length=n) |> collect)

function OpenMDAOCore.setup(self::ECompWithLargeOption)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompWithLargeOption, inputs, outputs)
    outputs["y"][1] = 2*self.a[3]*inputs["x"][1]^2 + 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompWithLargeOption, inputs, partials)
    partials["y", "x"][1] = 4*self.a[3]*inputs["x"][1]
    return nothing
end

struct ECompMatrixFree <: OpenMDAOCore.AbstractExplicitComp
    nrows::Int
    ncols::Int
end

function OpenMDAOCore.setup(self::ECompMatrixFree)
    input_data = [VarData("x1"; shape=(2, 3)), VarData("x2"; shape=(self.nrows, self.ncols))]
    output_data = [VarData("y1"; shape=(2, 3)), VarData("y2"; shape=(self.nrows, self.ncols))]
    partials_data = [PartialsData("*", "*")]  # I think this should work.

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompMatrixFree, inputs, outputs)
    x1, x2 = inputs["x1"], inputs["x2"]
    y1, y2 = outputs["y1"], outputs["y2"]
    @. y1 = 2*x1 + 3*x2^2
    @. y2 = 4*x1^3 + 5*x2^4
    return nothing
end

function OpenMDAOCore.compute_jacvec_product!(self::ECompMatrixFree, inputs, d_inputs, d_outputs, mode)
    x1, x2 = inputs["x1"], inputs["x2"]
    x1dot = get(d_inputs, "x1", nothing)
    x2dot = get(d_inputs, "x2", nothing)
    y1dot = get(d_outputs, "y1", nothing)
    y2dot = get(d_outputs, "y2", nothing)
    if mode == "fwd"
        # For forward mode, we are tracking the derivatives of everything with
        # respect to upstream inputs, and our goal is to calculate the
        # derivatives of this components outputs wrt the upstream inputs given
        # the derivatives of inputs wrt the upstream inputs.
        if y1dot !== nothing
            fill!(y1dot, 0)
            if x1dot !== nothing
                @. y1dot += 2*x1dot
            end
            if x2dot !== nothing
                @. y1dot += 6*x2*x2dot
            end
        end
        if y2dot !== nothing
            fill!(y2dot, 0)
            if x1dot !== nothing
                @. y2dot += 12*x1^2*x1dot
            end
            if x2dot !== nothing
                @. y2dot += 20*x2^3*x2dot
            end
        end
    elseif mode == "rev"
        # For reverse mode, we are tracking the derivatives of everything with
        # respect to a downstream output, and our goal is to calculate the
        # derivatives of the downstream output wrt each input given the
        # derivatives of the downstream output wrt each output.
        #
        # So, let's say I have a function f(y1, y2).
        # I start with fdot = df/df = 1.
        # Then I say that y1dot = df/dy1 = fdot*df/dy1
        # and y2dot = df/dy2 = fdot*df/dy2
        # Hmm...
        # f(y1(x1,x2), y2(x1, x2)) = df/dy1*(dy1/dx1 + dy1/dx2) + df/dy2*(dy2/dx1 + dy2/dx2)
        if x1dot !== nothing
            fill!(x1dot, 0)
            if y1dot !== nothing
                @. x1dot += y1dot*2
            end
            if x2dot !== nothing
                @. x1dot += y2dot*(12*x1^2)
            end
        end
        if x2dot !== nothing
            fill!(x2dot, 0)
            if y1dot !== nothing
                @. x2dot += y1dot*(6*x2)
            end
            if y2dot !== nothing
                @. x2dot += y2dot*(20*x2^3)
            end
        end
    end
    return nothing
end

struct ECompShapeByConn <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::ECompShapeByConn)
    input_data = [VarData("x"; shape_by_conn=true)]
    output_data = [VarData("y"; shape_by_conn=true, copy_shape="x")]
    partials_data = []

    return input_data, output_data, partials_data
end

function OpenMDAOCore.setup_partials(self::ECompShapeByConn, input_sizes, output_sizes)
    @assert input_sizes["x"] == output_sizes["y"]
    m, n = input_sizes["x"]
    rows, cols = OpenMDAOCore.get_rows_cols(ss_sizes=Dict(:i=>m, :j=>n), of_ss=[:i, :j], wrt_ss=[:i, :j])
    partials_data = [PartialsData("y", "x"; rows=rows, cols=cols)]

    return self, partials_data
end

function OpenMDAOCore.compute!(self::ECompShapeByConn, inputs, outputs)
    x = inputs["x"]
    y = outputs["y"]
    y .= 2 .* x.^2 .+ 1
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompShapeByConn, inputs, partials)
    x = inputs["x"]
    m, n = size(x)
    # So, with the way I've declared the partials above, OpenMDAO will have
    # created a Numpy array of shape (m, n) and then flattened it. So, to get
    # that to work, I'll need to do this:
    dydx = PermutedDimsArray(reshape(partials["y", "x"], n, m), (2, 1))
    dydx .= 4 .* x
    return nothing
end

struct ECompDomainError <: OpenMDAOCore.AbstractExplicitComp end

function OpenMDAOCore.setup(self::ECompDomainError)
    input_data = [VarData("x")]
    output_data = [VarData("y")]
    partials_data = [PartialsData("y", "x")]

    return input_data, output_data, partials_data
end

function OpenMDAOCore.compute!(self::ECompDomainError, inputs, outputs)
    if real(inputs["x"][1]) < 0
        throw(DomainError(real(inputs["x"][1]), "x must be >= 0"))
    else
        outputs["y"][1] = 2*inputs["x"][1]^2 + 1
    end
    return nothing
end

function OpenMDAOCore.compute_partials!(self::ECompDomainError, inputs, partials)
    if inputs["x"][1] < 0
        throw(DomainError(x, "x must be >= 0"))
    else
        partials["y", "x"][1] = 4*inputs["x"][1]
    end
    return nothing
end

end # module
