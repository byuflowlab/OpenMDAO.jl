using OpenMDAOCore
using Test
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes

function f_simple!(Y, X, params)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]
    e = @view Y[:e]
    f = @view Y[:f]
    g = @view Y[:g]

    M, N = params
    for n in 1:N
        e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
        for m in 1:M
            f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
            g[n, m] = 10*sin(b[n])*cos(d[m, n])
        end
    end

    return nothing
end


function doit()
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    params_simple = (M, N)

    # Also need copies of X_ca and Y_ca.
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    X_ca[:a] = 2.0
    X_ca[:b] .= range(3.0, 4.0; length=N)
    X_ca[:c] .= range(5.0, 6.0; length=M)
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # Now we can create the component.
    comp = SimpleMatrixFreeForwardDiffExplicitComp(f_simple!, Y_ca, X_ca; params=params_simple)

    inputs_dict = ca2strdict(get_input_ca(comp))
    inputs_dict["a"] .= 2.0
    inputs_dict["b"] .= range(3.0, 4.0; length=N)
    inputs_dict["c"] .= range(5.0, 6.0; length=M)
    inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    outputs_dict = ca2strdict(get_output_ca(comp))

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict["e"] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict["f"] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict["g"] .≈ g_check)

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the inputs.
    dx = get_dinput_ca(comp)
    dx .= rand(length(dx))
    dinputs_dict = ca2strdict(dx)
    doutputs_dict = ca2strdict(get_doutput_ca(comp))
    OpenMDAOCore.compute_jacvec_product!(comp, inputs_dict, dinputs_dict, doutputs_dict, "fwd")

    # Hmm... so how do I check this?
    # Well, just got to do the matrix-vector product myself.
    de_check = similar(e_check)
    de_check .= 0.0

    # OK, so now I can get a's contribution to the derivative of e.
    # I need to think about deda_check having a size of (N, 1).
    # And is a scalar, and e has size (N,).
    # So then I need to multiply deda_check by dx[:a] and add that to de_check.
    # For deda, the derivative is this:
    deda_check = zeros(N, 1)
    deda_check[:, 1] .= 4*only(a)
    de_check .+= deda_check * dx[:a]

    # Next, the derivative of e wrt b.
    dedb_check = zeros(N, N)
    for n in 1:N
        dedb_check[n, n] = (3*2.1)*b[n]^1.1
    end
    de_check .+= dedb_check * dx[:b]

    # Hmm... will this work?
    # I think I'll flatten things to make sure.
    # Actually don't think that's necessary.
    dedc_check = zeros(N, M)
    for m in 1:M
        for n in 1:N
            dedc_check[n, m] = (4*2.2)*c[m]^1.2
        end
    end
    de_check .+= dedc_check * dx[:c]

    # Hmm... will this work?
    # I think I should reshape things.
    dedd_check = zeros(N, M, N)
    for n in 1:N
        for m in 1:M
            dedd_check[n, m, n] = (5*2.3)*d[m, n]^1.3
        end
    end
    dxd_rs = reshape(dx[:d], M*N)
    dedd_check_rs = reshape(dedd_check, N, M*N)
    de_check .+= dedd_check_rs * dxd_rs

    # Did all the inputs to `e`, so we're ready to test.
    @test all(doutputs_dict["e"] .≈ de_check)

    # Now do `f`.
    df_check = similar(f_check)
    df_check .= 0.0

    for m in 1:M
        for n in 1:N
            df_check[m, n] += (6*2.4)*only(a)^1.4 * only(dx[:a])
        end
    end

    for n in 1:N
        for m in 1:M
            df_check[m, n] += (7*2.5)*b[n]^1.5 * dx[:b][n]
        end
    end

    for n in 1:N
        for m in 1:M
            df_check[m, n] += (8*2.6)*c[m]^1.6 * dx[:c][m]
        end
    end

    for n in 1:N
        for m in 1:M
            df_check[m, n] += (9*2.7)*d[m, n]^1.7 * dx[:d][m, n]
        end
    end

    @test all(doutputs_dict["f"] .≈ df_check)

    # Now do `g`.
    dg_check = similar(g_check)
    dg_check .= 0.0

    # g is not a function of `a`, so skip that.

    # Derivative wrt b.
    for m in 1:M
        for n in 1:N
            dg_check[n, m] += 10*cos(b[n])*cos(d[m, n]) * dx[:b][n]
        end
    end

    # g is not a function of `c`, so skip that.

    # Derivative wrt d.
    for m in 1:M
        for n in 1:N
            dg_check[n, m] += -10*sin(b[n])*sin(d[m, n]) * dx[:d][m, n]
        end
    end

    @test all(doutputs_dict["g"] .≈ dg_check)
end
doit()
