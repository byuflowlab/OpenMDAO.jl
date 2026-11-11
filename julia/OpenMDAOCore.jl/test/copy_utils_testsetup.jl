@testsetup module CopyUtils

using BenchmarkTools: @benchmark
using ComponentArrays: ComponentVector, ComponentMatrix, getaxes, getdata
using OpenMDAOCore: OpenMDAOCore, _copy_component_inputs!,
    _copy_component_outputs!, _copy_component_outputs_accum!,
    _copy_dense_partials!, _assemble_sparse_partials!, SparsePartialBlock
using SparseArrays: sparse
using SparseMatrixColorings
using Test: @test, @allocated

function sparse_extension()
    extension = Base.get_extension(
        OpenMDAOCore,
        :OpenMDAOCoreSparseMatrixColoringsExt,
    )
    @test extension !== nothing
    return extension
end

# The "old" (key-based) copying implementations, for comparison.

function old_copy_inputs!(data, inputs)
    for name in keys(data)
        @view(data[name]) .= inputs[string(name)]
    end
    return nothing
end

function old_copy_outputs!(data, outputs)
    for name in keys(data)
        name_string = string(name)
        if outputs[name_string] isa AbstractArray
            outputs[name_string] .= @view(data[name])
        else
            outputs[name_string] = only(@view(data[name]))
        end
    end
    return nothing
end

function old_copy_outputs_accum!(data, outputs)
    for name in keys(data)
        name_string = string(name)
        if outputs[name_string] isa AbstractArray
            outputs[name_string] .+= @view(data[name])
        else
            outputs[name_string] += only(@view(data[name]))
        end
    end
    return nothing
end

function old_dense_partials!(partials, J_ca)
    raxis, caxis = getaxes(J_ca)
    for oname in keys(raxis)
        ostr = string(oname)
        for iname in keys(caxis)
            istr = string(iname)
            Jsub_in = @view(J_ca[oname, iname])
            local Jsub_out
            try
                Jsub_out = partials[ostr, istr]
            catch e
                isa(e, KeyError) || rethrow()
            else
                Jsub_out .= reshape(Jsub_in, size(Jsub_out))
            end
        end
    end
    return nothing
end

function old_sparse_assembly!(J_ca, partials, raxis, caxis, rcdict)
    for oname in keys(raxis)
        ostr = string(oname)
        for iname in keys(caxis)
            istr = string(iname)
            Jsub_in = @view(J_ca[oname, iname])
            nrows = length(raxis[oname])
            ncols = length(caxis[iname])
            Jsub_in_reshape = reshape(Jsub_in, nrows, ncols)
            rows, cols = rcdict[oname, iname]
            local Jsub_out
            try
                Jsub_out = partials[ostr, istr]
            catch e
                isa(e, KeyError) || rethrow()
            else
                Jsub_out_vec = OpenMDAOCore._maybe_nonzeros(Jsub_out)
                Jsub_out_vec .= getindex.(Ref(Jsub_in_reshape), rows, cols)
            end
        end
    end
    return nothing
end

export sparse_extension, old_copy_inputs!, old_copy_outputs!, old_copy_outputs_accum!,
       old_dense_partials!, old_sparse_assembly!, _copy_component_inputs!,
       _copy_component_outputs!, _copy_dense_partials!, _assemble_sparse_partials!,
       SparsePartialBlock, @benchmark,
       _copy_component_outputs_accum!, ComponentVector, ComponentMatrix,
       getaxes, getdata, sparse, @allocated

end # @testsetup CopyUtils
