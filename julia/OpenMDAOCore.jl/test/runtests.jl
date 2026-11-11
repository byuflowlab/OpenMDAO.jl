using SysInfo: SysInfo
using ReTestItems
# Load the weakdeps before `OpenMDAOCore` so the
# `OpenMDAOCoreSparseMatrixColoringsExt` extension is triggered on load and its
# re-exported names (`PerturbedDenseSparsityDetector`,
# `get_rows_cols_dict`, `get_rows_cols_dict_from_sparsity`, `ca2strdict_sparse`)
# are available to the test setup module.
# using SparseArrays
# using SparseMatrixColorings
using OpenMDAOCore: OpenMDAOCore

runtests(
    OpenMDAOCore;
    nworkers = max(1, SysInfo.ncores() - 1),
    # nworkers = SysInfo.ncores(),
    nworker_threads = 1,
    testitem_timeout = 600,
)
