module WTFZygoteFirstTry

using ADTypes: ADTypes
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using Zygote: Zygote
using DifferentiationInterface: DifferentiationInterface
using Random: rand!


function f_simple!(Y, X)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]
    e = @view Y[:e]
    f = @view Y[:f]
    g = @view Y[:g]

    M = length(c)
    N = length(b)
    for n in 1:N
        e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
        for m in 1:M
            f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
            g[n, m] = 10*sin(b[n])*cos(d[m, n])
        end
    end

    return nothing
end

function f_simple(X)
    @show size(X)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]
    # e = @view Y[:e]
    # f = @view Y[:f]
    # g = @view Y[:g]

    # M = length(c)
    # N = length(b)
    # for n in 1:N
    #     e[n] = 2*a^2 + 3*b[n]^2.1 + 4*sum(c.^2.2) + 5*sum((@view d[:, n]).^2.3)
    #     for m in 1:M
    #         f[m, n] = 6*a^2.4 + 7*b[n]^2.5 + 8*c[m]^2.6 + 9*d[m, n]^2.7
    #         g[n, m] = 10*sin(b[n])*cos(d[m, n])
    #     end
    # end
    e = (2*a^2) .+ 3.0.*b.^2.1 .+ 4.0.*sum(c.^2.2) .+ 5.0.*vec(sum(d.^2.3; dims=1))
    f = (6*a^2.4) .+ 7.0.*reshape(b, 1, :).^2.5 .+ 8.0.*c.^2.6 .+ 9.0.*d.^2.7
    g = 10.0.*sin.(b).*cos.(PermutedDimsArray(d, (2, 1)))

    Y = ComponentVector(e=e, f=f, g=g)
    @show size(Y)
    return Y
end

function f_simple_zygote_compat(X)
    M = length(X.c)
    N = length(X.b)
    TF = eltype(X)
    Y = Zygote.Buffer(ComponentVector(e=zeros(TF, N), f=zeros(TF, M, N), g=zeros(TF, N, M)))
    f_simple!(Y, X)
    return Y
end

function doit()
    N = 3
    M = 4
    X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    # Y_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))
    ad_backend = ADTypes.AutoZygote()

    # dX_ca = similar(X_ca)
    # dY_ca = similar(Y_ca)
    dY_ca = ComponentVector(e=zeros(Float64, N), f=zeros(Float64, M, N), g=zeros(Float64, N, M))

    # prep = DifferentiationInterface.prepare_pullback(f_simple!, Y_ca, ad_backend, X_ca, (dY_ca,))
    # prep = DifferentiationInterface.prepare_pullback(f_simple, ad_backend, getdata(X_ca), (getdata(dY_ca),))
    prep = DifferentiationInterface.prepare_pullback(f_simple, ad_backend, X_ca, (dY_ca,))
    # prep = DifferentiationInterface.prepare_pullback(f_simple_zygote_compat, ad_backend, getdata(X_ca), (getdata(dY_ca),))

    X_ca.a = 2.0
    X_ca.b .= range(3.0, 4.0; length=N)
    X_ca.c .= range(5.0, 6.0; length=M)
    X_ca.d .= reshape(range(7.0, 8.0; length=M*N), M, N)

    rand!(dY_ca)

    dX_ca = DifferentiationInterface.pullback(f_simple, prep, ad_backend, X_ca, (dY_ca,))
    # dX_ca = DifferentiationInterface.pullback(f_simple_zygote_compat, prep, ad_backend, X_ca, (dY_ca,))

    return dX_ca
end


end # module
