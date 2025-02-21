using OpenMDAOCore
using Test
using ADTypes: ADTypes
using ComponentArrays: ComponentVector, ComponentMatrix, getdata, getaxes
using ForwardDiff: ForwardDiff
using Enzyme: Enzyme
using EnzymeCore: EnzymeCore
using ReverseDiff: ReverseDiff
using Zygote: Zygote

function f_simple(X, params)
    a = only(X[:a])
    b = @view X[:b]
    c = @view X[:c]
    d = @view X[:d]

    e = (2*a^2) .+ 3.0.*b.^2.1 .+ 4.0.*sum(c.^2.2) .+ 5.0.*vec(sum(d.^2.3; dims=1))
    f = (6*a^2.4) .+ 7.0.*reshape(b, 1, :).^2.5 .+ 8.0.*c.^2.6 .+ 9.0.*d.^2.7
    g = 10.0.*sin.(b).*cos.(PermutedDimsArray(d, (2, 1)))

    Y = ComponentVector(e=e, f=f, g=g)
    return Y
end

function doit_forward(; ad_type, disable_prep)
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    params = (M, N)

    # Also need copies of X_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    X_ca = ComponentVector(a=zero(Float64), d=zeros(Float64, M, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"ft", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"in**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zeros(Float64, M)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"psi", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M)))
    aviary_input_names = Dict(
                            :a=>"foo:bar:baz:a",
                            :b=>"foo:bar:baz:b",
                            :c=>"foo:bar:baz:c",
                            :d=>"foo:bar:baz:d")
    aviary_output_names = Dict(
                            :e=>"foo:bar:baz:e",
                            :f=>"foo:bar:baz:f",
                            :g=>"foo:bar:baz:g")

    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    # X_ca[:a] = 2.0
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0*(1/0.0254)*(1/12)
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N) .* 0.0254^2
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M) .* 2.23694
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # Have the inputs a, b, and c set to the default values in the aviary metadata, but convert them from the default units to specified units.
    # Also make sure output e has different units than default.
    units_dict = Dict(:a=>"m", :b=>"m**2", :c=>"m/s", :e=>"Pa")

    # Now we can create the component.
    if ad_type == "forwarddiff"
        ad_backend = ADTypes.AutoForwardDiff()
    elseif ad_type == "enzymeforward"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Forward)
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, disable_prep, units_dict, aviary_input_names, aviary_output_names, aviary_meta_data)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # inputs_dict["a"] .= 2.0
    # inputs_dict["b"] .= range(3.0, 4.0; length=N)
    # inputs_dict["c"] .= range(5.0, 6.0; length=M)
    # inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    # outputs_dict = ca2strdict(f_simple(get_input_ca(comp), params))
    outputs_dict = ca2strdict(f_simple(get_input_ca(comp), params), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    # a, b, c, d = getindex.(Ref(inputs_dict), ["a", "b", "c", "d"])
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the inputs.
    dx = get_dinput_ca(comp)
    dx .= rand(length(dx))
    dinputs_dict = ca2strdict(dx, aviary_input_names)
    doutputs_dict = ca2strdict(get_doutput_ca(comp), aviary_output_names)
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
    @test all(doutputs_dict[get_aviary_output_name(comp, :e)] .≈ de_check)

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

    @test all(doutputs_dict[get_aviary_output_name(comp, :f)] .≈ df_check)

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

    @test all(doutputs_dict[get_aviary_output_name(comp, :g)] .≈ dg_check)
end

doit_forward(; disable_prep=false, ad_type="forwarddiff")

# Fails with
#
# out-of-place: Error During Test at /home/dingraha/.julia/packages/SafeTestsets/raUNr/src/SafeTestsets.jl:30
#   Got exception outside of a @test
#   LoadError: MethodError: no method matching similar(::DifferentiationInterface.NoPushforwardPrep, ::Type{ForwardDiff.Dual{ForwardDiff.Tag{Op
# enMDAOCore.var"#35#36"{typeof(Main.var"##out-of-place#231".f_simple), Tuple{Int64, Int64}}, Float64}, Any, 1}})
#   The function `similar` exists, but no method is defined for this combination of argument types.
  
#   Closest candidates are:
#     similar(::Type{A}, ::Type{T}, ::StaticArraysCore.Size{S}) where {A<:Array, T, S}
#      @ StaticArrays ~/.julia/packages/StaticArrays/2dZx4/src/abstractarray.jl:136
#     similar(::Type{SA}, ::Type{T}, ::StaticArraysCore.Size{S}) where {SA<:StaticArraysCore.SizedArray, T, S}
#      @ StaticArrays ~/.julia/packages/StaticArrays/2dZx4/src/abstractarray.jl:135
#     similar(::Type{SA}, ::Type{T}) where {SA<:StaticArraysCore.StaticArray, T}
#      @ StaticArrays ~/.julia/packages/StaticArrays/2dZx4/src/abstractarray.jl:120
#     ...
  
#   Stacktrace:
#     [1] make_dual_similar(::Type{ForwardDiff.Tag{OpenMDAOCore.var"#35#36"{typeof(Main.var"##out-of-place#231".f_simple), Tuple{Int64, Int64}}
# , Float64}}, x::DifferentiationInterface.NoPushforwardPrep, tx::Tuple{ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{Compone
# ntArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}
# }})
#       @ DifferentiationInterfaceForwardDiffExt ~/projects/aviary_propeller/dev/aviary_bemt_propeller/aviary_bemt_propeller/julia/AviaryBEMTPr
# opellerComponents.jl/dev/OpenMDAO/julia/OpenMDAOCore.jl/dev/DifferentiationInterface/DifferentiationInterface/ext/DifferentiationInterfaceFor
# wardDiffExt/utils.jl:34
#     [2] prepare_pushforward(::OpenMDAOCore.var"#35#36"{typeof(Main.var"##out-of-place#231".f_simple), Tuple{Int64, Int64}}, ::Differentiation
# Interface.NoPushforwardPrep, ::ADTypes.AutoForwardDiff{nothing, Nothing}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{C
# omponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3
# ))))}}}, ::Tuple{ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis(
# (3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}})
#       @ DifferentiationInterfaceForwardDiffExt ~/projects/aviary_propeller/dev/aviary_bemt_propeller/aviary_bemt_propeller/julia/AviaryBEMTPr
# opellerComponents.jl/dev/OpenMDAO/julia/OpenMDAOCore.jl/dev/DifferentiationInterface/DifferentiationInterface/ext/DifferentiationInterfaceFor
# wardDiffExt/twoarg.jl:13
#     [3] value_and_pushforward(::OpenMDAOCore.var"#35#36"{typeof(Main.var"##out-of-place#231".f_simple), Tuple{Int64, Int64}}, ::Differentiati
# onInterface.NoPushforwardPrep, ::ADTypes.AutoForwardDiff{nothing, Nothing}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple
# {ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4,
#  3))))}}}, ::Tuple{ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxi
# s((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}})
#       @ DifferentiationInterface ~/projects/aviary_propeller/dev/aviary_bemt_propeller/aviary_bemt_propeller/julia/AviaryBEMTPropellerCompone
# nts.jl/dev/OpenMDAO/julia/OpenMDAOCore.jl/dev/DifferentiationInterface/DifferentiationInterface/src/fallbacks/no_prep.jl:200
#     [4] value_and_pushforward!(::OpenMDAOCore.var"#35#36"{typeof(Main.var"##out-of-place#231".f_simple), Tuple{Int64, Int64}}, ::Tuple{Compon
# entArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(e = ViewAxis(1:3, Shaped1DAxis((3,))), f = ViewAxis(4:15, Sha
# pedAxis((4, 3))), g = ViewAxis(16:27, ShapedAxis((3, 4))))}}}}, ::DifferentiationInterface.NoPushforwardPrep, ::ADTypes.AutoForwardDiff{nothi
# ng, Nothing}, ::ComponentArrays.ComponentVector{Float64, Vector{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((
# 3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(9:20, ShapedAxis((4, 3))))}}}, ::Tuple{ComponentArrays.ComponentVector{Float64, Ve
# ctor{Float64}, Tuple{ComponentArrays.Axis{(a = 1, b = ViewAxis(2:4, Shaped1DAxis((3,))), c = ViewAxis(5:8, Shaped1DAxis((4,))), d = ViewAxis(
# 9:20, ShapedAxis((4, 3))))}}}})
#       @ DifferentiationInterface ~/projects/aviary_propeller/dev/aviary_bemt_propeller/aviary_bemt_propeller/julia/AviaryBEMTPropellerCompone
# nts.jl/dev/OpenMDAO/julia/OpenMDAOCore.jl/dev/DifferentiationInterface/DifferentiationInterface/src/first_order/pushforward.jl:207
# doit_forward(; disable_prep=true, ad_type="forwarddiff")

# Fails with
  # LoadError: UndefRefError: access to undefined reference
  # Stacktrace:
  #   [1] LLVM.Value(ref::Ptr{LLVM.API.LLVMOpaqueValue})
  #     @ LLVM ~/.julia/packages/LLVM/b3kFs/src/core/value.jl:39
  #   [2] jl_nthfield_fwd
  #     @ ~/.julia/packages/Enzyme/QsaeA/src/rules/typeunstablerules.jl:1554 [inlined]
  #   [3] jl_nthfield_fwd_cfunc(B::Ptr{LLVM.API.LLVMOpaqueBuilder}, OrigCI::Ptr{LLVM.API.LLVMOpaqueValue}, gutils::Ptr{Nothing}, normalR::Ptr{P
# tr{LLVM.API.LLVMOpaqueValue}}, shadowR::Ptr{Ptr{LLVM.API.LLVMOpaqueValue}})
  #     @ Enzyme.Compiler ~/.julia/packages/Enzyme/QsaeA/src/rules/llvmrules.jl:75
  #   [4] EnzymeCreateForwardDiff(logic::Enzyme.Logic, todiff::LLVM.Function, retType::Enzyme.API.CDIFFE_TYPE, constant_args::Vector{Enzyme.API
# .CDIFFE_TYPE}, TA::Enzyme.TypeAnalysis, returnValue::Bool, mode::Enzyme.API.CDerivativeMode, runtimeActivity::Bool, width::Int64, additionalA
# rg::Ptr{Nothing}, typeInfo::Enzyme.FnTypeInfo, uncacheable_args::Vector{Bool})
  #     @ Enzyme.API ~/.julia/packages/Enzyme/QsaeA/src/api.jl:334
# doit_forward(; disable_prep=false, ad_type="enzymeforward")


function doit_reverse(; ad_type, disable_prep)
    # `M` and `N` will be passed via the params argument.
    N = 3
    M = 4
    params = (M, N)

    # Also need copies of X_ca and Y_ca.
    # X_ca = ComponentVector(a=zero(Float64), b=zeros(Float64, N), c=zeros(Float64, M), d=zeros(Float64, M, N))
    X_ca = ComponentVector(d=zeros(Float64, M, N))
    aviary_meta_data = Dict(
                            "foo:bar:baz:a"=>Dict("units"=>"ft", "default_value"=>zero(Float64)),
                            "foo:bar:baz:b"=>Dict("units"=>"in**2", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:c"=>Dict("units"=>"mi/hr", "default_value"=>zeros(Float64, M)),
                            "foo:bar:baz:d"=>Dict("units"=>"kg", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:e"=>Dict("units"=>"psi", "default_value"=>zeros(Float64, N)),
                            "foo:bar:baz:f"=>Dict("units"=>"kg/m**3", "default_value"=>zeros(Float64, M, N)),
                            "foo:bar:baz:g"=>Dict("units"=>"s", "default_value"=>zeros(Float64, N, M)))
    aviary_input_names = Dict(
                            :a=>"foo:bar:baz:a",
                            :b=>"foo:bar:baz:b",
                            :c=>"foo:bar:baz:c",
                            :d=>"foo:bar:baz:d")
    aviary_output_names = Dict(
                            :e=>"foo:bar:baz:e",
                            :f=>"foo:bar:baz:f",
                            :g=>"foo:bar:baz:g")


    # Need to fill `X_ca` with "reasonable" values for the sparsity detection stuff to work.
    # X_ca[:a] = 2.0
    aviary_meta_data[aviary_input_names[:a]]["default_value"] = 2.0*(1/0.0254)*(1/12)
    # X_ca[:b] .= range(3.0, 4.0; length=N)
    # X_ca[:c] .= range(5.0, 6.0; length=M)
    aviary_meta_data[aviary_input_names[:b]]["default_value"] .= range(3.0, 4.0; length=N) .* 0.0254^2
    aviary_meta_data[aviary_input_names[:c]]["default_value"] .= range(5.0, 6.0; length=M) .* 2.23694
    X_ca[:d] .= reshape(range(7.0, 8.0; length=M*N), M, N)

    # Have the inputs a, b, and c set to the default values in the aviary metadata, but convert them from the default units to specified units.
    # Also make sure output e has different units than default.
    units_dict = Dict(:a=>"m", :b=>"m**2", :c=>"m/s", :e=>"Pa")

    # Now we can create the component.
    if ad_type == "reversediff"
        ad_backend = ADTypes.AutoReverseDiff()
    elseif ad_type == "enzymereverse"
        ad_backend = ADTypes.AutoEnzyme(; mode=EnzymeCore.Reverse)
    elseif ad_type == "zygote"
        ad_backend = ADTypes.AutoZygote()
    else
        error("unexpected ad_type = $(ad_type)")
    end
    comp = MatrixFreeADExplicitComp(ad_backend, f_simple, X_ca; params, disable_prep, units_dict, aviary_input_names, aviary_output_names, aviary_meta_data)

    # Need to make sure the units are what I expect them to be.
    @test get_units(comp, :a) == "m"
    @test get_units(comp, :b) == "m**2"
    @test get_units(comp, :c) == "m/s"
    @test get_units(comp, :d) == "kg"
    @test get_units(comp, :e) == "Pa"
    @test get_units(comp, :f) == "kg/m**3"
    @test get_units(comp, :g) == "s"

    # inputs_dict = ca2strdict(get_input_ca(comp))
    # inputs_dict["a"] .= 2.0
    # inputs_dict["b"] .= range(3.0, 4.0; length=N)
    # inputs_dict["c"] .= range(5.0, 6.0; length=M)
    # inputs_dict["d"] .= reshape(range(7.0, 8.0; length=M*N), M, N)
    inputs_dict = ca2strdict(get_input_ca(comp), aviary_input_names)
    outputs_dict = ca2strdict(get_output_ca(comp), aviary_output_names)

    OpenMDAOCore.compute!(comp, inputs_dict, outputs_dict)
    a, b, c, d = getindex.(Ref(inputs_dict), [get_aviary_input_name(comp, :a), get_aviary_input_name(comp, :b), get_aviary_input_name(comp, :c), get_aviary_input_name(comp, :d)])
    e_check = 2.0*a.^2 .+ 3 .* b.^2.1 .+ 4*sum(c.^2.2) .+ 5 .* sum(d.^2.3; dims=1)[:]
    @test all(outputs_dict[get_aviary_output_name(comp, :e)] .≈ e_check)

    f_check = 6.0*a.^2.4 .+ 7 .* reshape(b, 1, :).^2.5 .+ 8 .* c.^2.6 .+ 9 .* d.^2.7
    @test all(outputs_dict[get_aviary_output_name(comp, :f)] .≈ f_check)

    g_check = 10 .* sin.(b).*cos.(transpose(d))
    @test all(outputs_dict[get_aviary_output_name(comp, :g)] .≈ g_check)

    # So, to call `_compute_jacvec_product!`, I need a dict of derivatives that, I think, is like the outputs.
    dy = get_doutput_ca(comp)
    dy .= rand(length(dy))
    doutputs_dict = ca2strdict(dy, aviary_output_names)
    dinputs_dict = ca2strdict(get_dinput_ca(comp), aviary_input_names)
    OpenMDAOCore.compute_jacvec_product!(comp, inputs_dict, dinputs_dict, doutputs_dict, "rev")

    # Hmm... so how do I check this?
    # Well, just got to do the vector-jacobian product myself.
    da_check = similar(a)
    da_check .= 0.0

    # First, derivative of e wrt a.
    for n in 1:N
        da_check[1] += dy[:e][n] * 4*only(a)
    end

    # Next, derivative of f wrt a.
    for m in 1:M
        for n in 1:N
            da_check[1] += dy[:f][m, n] * (6*2.4)*only(a)^1.4
        end
    end

    # Derivative of g wrt a is zero.

    # Did all the outputs with `a`, so we're ready to test.
    @test all(dinputs_dict[get_aviary_input_name(comp, :a)] .≈ da_check)

    # Next, `b`.
    db_check = similar(b)
    db_check .= 0.0

    # First do the derivative of e wrt b.
    for n in 1:N
        db_check[n] += dy[:e][n] * (3*2.1)*b[n]^1.1
    end

    # Next, derivative of f wrt b.
    for n in 1:N
        for m in 1:M
            db_check[n] += dy[:f][m, n] * (7*2.5)*b[n]^1.5
        end
    end

    # Next, derivative of g wrt b.
    for m in 1:M
        for n in 1:N
            db_check[n] += dy[:g][n, m] * 10*cos(b[n])*cos(d[m, n])
        end
    end

    # That's all the outputs with `b`, so we're ready to check.
    @test all(dinputs_dict[get_aviary_input_name(comp, :b)] .≈ db_check)

    # Now derivatives wrt c.
    dc_check = similar(c)
    dc_check .= 0.0

    # Derivative of `e` wrt c.
    for m in 1:M
        for n in 1:N
            dc_check[m] += dy[:e][n] * (4*2.2)*c[m]^1.2
        end
    end

    # Derivative of `f` wrt c.
    for n in 1:N
        for m in 1:M
            dc_check[m] += dy[:f][m, n] * (8*2.6)*c[m]^1.6
        end
    end

    # Derivative of `g` wrt c is 0.
    
    # Did all the outputs, so ready to check.
    @test all(dinputs_dict[get_aviary_input_name(comp, :c)] .≈ dc_check)

    # Now derivatives wrt d.
    dd_check = similar(d)
    dd_check .= 0.0

    # Derivative of e wrt d.
    for n in 1:N
        for m in 1:M
            dd_check[m, n] += dy[:e][n] * (5*2.3)*d[m, n]^1.3
        end
    end

    # Derivative of f wrt d.
    for n in 1:N
        for m in 1:M
            dd_check[m, n] += dy[:f][m, n] * (9*2.7)*d[m, n]^1.7 
        end
    end

    # Derivative of g wrt d.
    for m in 1:M
        for n in 1:N
            dd_check[m, n] += dy[:g][n, m] * (-10)*sin(b[n])*sin(d[m, n])
        end
    end

    # Now check.
    @test all(dinputs_dict[get_aviary_input_name(comp, :d)] .≈ dd_check)

    return nothing
end

doit_reverse(; ad_type="reversediff", disable_prep=false)
doit_reverse(; ad_type="reversediff", disable_prep=true)

# Fails with scary segfault:
# doit_reverse(; ad_type="enzymereverse", disable_prep=false)

doit_reverse(; ad_type="zygote", disable_prep=false)
doit_reverse(; ad_type="zygote", disable_prep=true)
