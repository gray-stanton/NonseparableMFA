using LinearAlgebra
using BSplines
using Statistics
using OffsetArrays

# Structure of MFA parameters and associated

struct ChannelSpec
    C :: Int
    Ncs :: Vector{Int} # Sizes of the Channels (Ncs[i]>0), length is C
    N :: Int # Total number of observations, == sum(Ncs)
    r0 :: Int # Number of Common Factors
    rcs :: Vector{Int} # Number of unique factors per channel, length is C
    r :: Int # Total number of unique factors
end
ChannelSpec(Ncs, rcs, r0) = ChannelSpec(length(Ncs), Ncs, sum(Ncs), r0, rcs, sum(rcs))

# Spline representation of matrix function L(w): [a, b] -> LowerTriangular
struct MFACholeskyBSplineFunc{S<:Complex}
    bs :: BSplineBasis # B-Spline basis for cholesky function
    blockdims :: Vector{Int} # Row/col size of matrix function subblocks
    totdim  :: Int # Total row/col size of matrix function
    coeffs :: Vector{S} # Packed B-spline coefficients, Spl-> Chan -> col -> row
end
MFACholeskyBSplineFunc(bs :: BSplineBasis, blockdims :: Vector{Int}, paramtype=ComplexF64) = MFACholeskyBSplineFunc{paramtype}(
    bs,
    blockdims,
    sum(blockdims),
    zeros(paramtype, length(bs)* sum([s*(s+1)÷2 for s in blockdims]))
)

struct MFASpecDensBSplineFunc{S<:Complex}
    L :: MFACholeskyBSplineFunc{S}
end

# Parameter object for MFA with dependent latent factors
mutable struct MFADependent{T<:Real, S<:Complex}
    A :: Matrix{T} # N x r_0 common factor loadings
    B :: Matrix{T} # N x r  block-diagonal unique factor loadings
    Bcs :: Array{Matrix{T}, 1} # C-length array of unique-factor loading blocks
    C :: Matrix{T} # N x (r_0 +r) combined factor loadings, == [A B]
    P :: Diagonal{T, Vector{T}} # N x N Diagonal matrix of idiosyncratic variances
    L :: MFACholeskyBSplineFunc{S} # B-spline rep. of factor spectral density cholesky function
    cs :: ChannelSpec # Specification of channel/factor structure
end
MFADependent(cs :: ChannelSpec, bs :: BSplineBasis, realparamtype=Float64, complexparamtype=ComplexF64) = MFADependent(
    zeros(realparamtype, cs.N, cs.r0), # A
    zeros(realparamtype, cs.N, cs.r), # B
    [zeros(realparamtype, Nc, rc) for (Nc, rc) in zip(cs.Ncs, cs.rcs)], # Bcs
    zeros(realparamtype, cs.N, cs.r0 + cs.r),
    Diagonal(zeros(realparamtype, cs.N)), # P
    MFACholeskyBSplineFunc(bs, vcat([cs.r0], cs.rcs), complexparamtype),
    cs
)


function blockdiag!(store, mats)
    ns = [size(mat)[1] for mat in mats]
    ms = [size(mat)[2] for mat in mats]
    n, m = size(store) # required that n = sum(ns), m = sum(ms) to fit_L
    ridx = 0
    cidx = 0
    for (c, mat) in enumerate(mats)
        store[(ridx+1):(ridx+ns[c]), (cidx+1):(cidx+ms[c])] = mat
        ridx += ns[c]
        cidx += ms[c]
    end
end

function blockdiag(mats)
    ns = [size(mat)[1] for mat in mats]
    ms = [size(mat)[2] for mat in mats]
    n, m = sum(ns), sum(ms)
    store = zeros(eltype(eltype(mats)), n, m)
    blockdiag!(store, mats)
    return store
end

function unblockdiag!(store, mat, blkrowsizes, blkcolsizes)
    C = length(blkrowsizes)
    ridx = 0
    cidx = 0
    for c in 1:C
        store[c][:] = mat[(ridx+1):(ridx+blkrowsizes[c]), (cidx+1):(cidx+blkcolsizes[c])]
        ridx += blkrowsizes[c]
        cidx += blkcolsizes[c]
    end
end

function unblockdiag(mat, blkrowsizes, blkcolsizes)
    store = [zeros(eltype(mat), n, m) for (n, m) in zip(blkrowsizes, blkcolsizes)]
    unblockdiag!(store, mat, blkrowsizes, blkcolsizes)
    return store
end

function halfvec!(store, mat)
    n, m = size(mat)
    idx = 0
    for j in 1:m
        for i in j:n
            idx+=1
            store[idx] = mat[i, j]
        end
    end
end


function halfvec(mat)
    n, m = size(mat)
    sz = n*min(n, m) - min(n, m) *  (min(n, m)-1) ÷ 2 # n + n-1 + ... + n-min(n,m)+1
    store = zeros(eltype(mat), sz)
    halfvec!(store, mat)
    return store
end


function unhalfvec!(store, vec, rowsize, colsize)
    idx = 0
    for j in 1:colsize
        for i in j:rowsize
            idx+=1
            store[i, j] = vec[idx]
        end
    end
end

function unhalfvec(vec, rowsize, colsize)
    store = zeros(eltype(vec), rowsize, colsize)
    unhalfvec!(store, vec, rowsize, colsize)
    return store
end


# Pack a sequence of length(bs) lower-triangular matrices into a coeffs vector
function packLTsequence!(store, Ls, blockdims)
    K = length(Ls)
    idx = 0
    for k in 1:K
        blks = unblockdiag(Ls[k],blockdims, blockdims)
        vs = halfvec.(blks)
        flatv = vcat(vs...)
        store[(idx+1):(idx+length(flatv))] = flatv
        idx += length(flatv)
    end
end

function packLTsequence(Ls, blockdims)
    K = length(Ls)
    totr = sum([s*(s+1) ÷ 2 for s in blockdims])
    store = zeros(eltype(eltype(Ls)), K*totr)
    packLTsequence!(store, Ls, blockdims)
    return store
end

function unpackLTsequence!(store, alpha, blockdims)
    K = length(store)
    C = length(blockdims)
    totr = length(alpha) ÷ K
    kidx = 0
    for k in 1:K
        alphasec = alpha[(kidx+1):(kidx+totr)] # pull out alpha for basis func k
        chidx = 0
        Ls1 = Matrix{eltype(eltype(store))}[] # LT matrices on basis func k
        for c in 1:C
            chansz = blockdims[c] * (blockdims[c] + 1) ÷ 2
            push!(Ls1, unhalfvec(alphasec[(chidx+1):(chidx+chansz)], blockdims[c], blockdims[c]))
            chidx += chansz
        end
        blockdiag!(store[k], Ls1)
        kidx += totr
    end
end


function unpackLTsequence(alpha, blockdims, K)
    p = sum(blockdims)
    store = [zeros(eltype(alpha), p, p) for _ in 1:K]
    unpackLTsequence!(store, alpha, blockdims)
    return store
end

# Extract the coefficient on the kth basis func in channel c, column j row i
# c == 0 gives element in common factor loading
function getelemidx(L :: MFACholeskyBSplineFunc, k, c, j, i)
    c2 = c+1
    K = length(L.bs)
    if (i < j) | (i <= 0) | (j <= 0)
        throw(ArgumentError("Must access lower triangle, (i = $i, j = $j)"))
    elseif (k > K) | (k <= 0)
        throw(ArgumentError("Must access valid basis function coeff, (k = $k, K = $K)"))
    elseif c2 > length(L.blockdims) | (c2 <= 0)
        throw(ArgumentError("Must access valid block, (c = $c, C+1 = $(length(L.blockdims))"))
    elseif (j > L.blockdims[c2]) | (i > L.blockdims[c2])
        throw(ArgumentError("Must access valid matrix section, , (i = $i, j = $j, rc = $(L.blockdims[c2])))"))
    end
    # First to correct bs func, then to correct chan, then to col, then to row
    totLsize = length(L.coeffs) ÷ K # number of coefs on a given bs function
    prebasisoffset = (k-1) * totLsize
    preblksizes = L.blockdims[1:(c2-1)]
    preblkoffset = sum([(s*(s+1) ÷ 2) for s in preblksizes]) # == 0 if c==0
    coloffset =  (j-1) * L.blockdims[c2] - ((j-2)*(j-1) ÷ 2) # rc + (rc-1) + (rc -j+1)
    rowoffset = (i-j+1)
    return prebasisoffset + preblkoffset + coloffset + rowoffset
end

getelem(L :: MFACholeskyBSplineFunc, k, c, j, i) = L.coeffs[getelemidx(L, k, c, j, i)]
function setelem(s, L :: MFACholeskyBSplineFunc, k, c, j, i)
    L.coeffs[getelemidx(L, k, c, j, i)] = s
end
 

# Enforce that int_-pi^pi L(w)*L(w)' == I
# As L(-w) = conj(L(w)) by properties of spec dens functions
# equivalent to enforcing that int_0^pi 2*real(L(w)L(w)') == I
# technically should do this via B-spline prod integrals, but for now just do sum
function identityintegral_invariant!(C :: Matrix{<:Real}, L :: MFACholeskyBSplineFunc, ws)
    D = MFASpecDensBSplineFunc(L)
    str = zeros(eltype(L.coeffs), L.totdim, L.totdim)
    riem_int = zeros(eltype(L.coeffs), L.totdim, L.totdim)
    for w in ws
        vl = 2*real(D(str, w))
        riem_int += vl / (length(ws)*pi) # assume ws equally spaced
    end
    R = cholesky(riem_int).L
    # set C -> C*R, L(w) -> R^-1 * L(w) 
    C[:, :] = C * R
    Lts = unpackLTsequence(L.coeffs, L.blockdims, length(L.bs))
    Ltsnew = [inv(R) * Lts[k] for k in 1:length(L.bs)]
    packLTsequence!(L.coeffs, Ltsnew, L.blockdims)
end

# Enforce that the diagonal of the cholesky factor be real and positive.
function realdiag_invariant!(L :: MFACholeskyBSplineFunc)
    K = length(L.bs)
    C  = length(L.blockdims)
    for k in 1:K
        for c2 in 1:C
            for j in 1:L.blockdims[c2]
                idx = getelemidx(L, k, c2-1, j, j)
                v = L.coeffs[idx]
                realrotfactor = exp(-1*im*angle(v))
                for i in j:L.blockdims[c2]
                    idx2 = getelemidx(L, k, c2-1, j, i)
                    L.coeffs[idx2] *= realrotfactor
                end
            end
        end
    end
end


# Rotate a tall nxm matrix so that top mxm block is lower-triangular with positive diagonals.
function lowertri_invariant!(A :: Matrix{<:Real})
    n, m = size(A)
    if n < m
        throw(ArgumentError("A not tall (n=$n, m=$m)"))
    end
    A1 = A[1:m, 1:m]
    A1LQ = lq(A1)
    # Rotate
    A[:, :] = A * A1LQ.Q'
    # Assert sign orientation
    for i in 1:m
        if sign(A[i, i]) == -1
            A[:,i] = -1*A[:, i]
        end
    end
end

function extractABcs(C, Ncs, r0, rcs)
    Astore = zeros(eltype(C), sum(Ncs), r0)
    Bcsstore = [zeros(eltype(C), Nc, rc) for (Nc, rc) in zip(Ncs, rcs)]
    extractABcs!(Astore, Bcsstore, C)
    return (Astore, Bcsstore)
end

function extractABcs!(Astore, Bcsstore, C :: Matrix{<:Real})
    Astore[:, :] = C[:, 1:(size(Astore)[2])]
    rowblksizes = [size(Bc)[1] for Bc in Bcsstore]
    colblksizes = [size(Bc)[2] for Bc in Bcsstore]
    unblockdiag!(Bcsstore, C[:, r0:end], rowblksizes, colblksizes)
end

function lowertri_invariant(A :: Matrix{<:Real})
    Anew = deepcopy(A)
    lowertri_invariant!(Anew)
    return Anew
end

# call MFACholeskyBSplineFunc to evaluate at point w, store value in store 
function (L::MFACholeskyBSplineFunc)(store, w)
    bsplinevals = bsplines(L.bs, w)
    ltcoefs = unpackLTsequence(L.coeffs, L.blockdims, length(L.bs))
    if !isnothing(bsplinevals)
        for (k, v) in pairs(bsplinevals)
            store[:, :] += v * ltcoefs[k]
        end
    end
end


function (L::MFACholeskyBSplineFunc)(w)
    value = zeros(eltype(L.coeffs), L.totdim, L.totdim)
    L(value, w)
    return value
end



function (D::MFASpecDensBSplineFunc)(store, w)
    Lv = D.L(w)
    store[:, :] = Lv * Lv'
end


function (D::MFASpecDensBSplineFunc)(w)
    store = zeros(eltype(D.L.coeffs), D.L.totdim, D.L.totdim)
    D(store, w)
    return store
end