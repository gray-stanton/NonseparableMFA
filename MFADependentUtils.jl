using LinearAlgebra
using BSplines
using Statistics
using OffsetArrays

# Structure of MFA parameters and associated

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


function packmodel(model :: MFADependent)
    v = vcat(halfvec(model.A), [halfvec(Bc) for Bc in model.Bcs]..., diag(model.P), model.L.coeffs)
    return v
end

function unpackmodel(modelvec, cs, bs)
    model = MFADependent(cs, bs)
    Asz = cs.N * cs.r0 - cs.r0*(cs.r0-1) ÷ 2
    Bsz = sum([cs.Ncs[c] * cs.rcs[c] - cs.rcs[c]*(cs.rcs[c]-1) ÷ 2 for c in 1:cs.C])
    avec = modelvec[1:Asz]
    Bvec = modelvec[(Asz+1):(Asz+Bsz)]
    Bcsvecs = Vector{eltype(Bvec)}[]
    idx = 0
    for c in 1:cs.C
        Bcsz = cs.Ncs[c] * cs.rcs[c] - cs.rcs[c]*(cs.rcs[c]-1) ÷ 2
        push!(Bcsvecs, Bvec[(idx+1):(idx+Bcsz)])
        idx += Bcsz
    end
    Pvec = modelvec[(Asz+Bsz+1):(Asz+Bsz+cs.N)]
    Lcoeffs = modelvec[(Asz+Bsz+cs.N+1):end]
    A = real.(unhalfvec(avec, cs.N, cs.r0))
    Bcs = [real.(unhalfvec(Bcsvecs[c], cs.Ncs[c], cs.rcs[c])) for c in 1:cs.C]
    P = Diagonal(real.(Pvec))
    B = blockdiag(Bcs)
    model.A = A
    model.B = B
    model.Bcs = Bcs
    model.C = hcat(A, B)
    model.P = P
    model.L = MFACholeskyBSplineFunc(bs, model.L.blockdims, model.L.totdim, Lcoeffs)
    return model
end

function factor_closest_rot(F0, F1)
    M = F1*F0'
    dec = svd(M)
    R = dec.U * dec.Vt
    return R
end

function specdens(model :: MFADependent, w)
    C = [model.A model.B]
    D = MFASpecDensBSplineFunc(model.L)(w)
    return C*D*C' + model.P
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
function getelemidx(blkdims, K, k, c, j, i)
    c2 = c+1
    if (i < j) | (i <= 0) | (j <= 0)
        throw(ArgumentError("Must access lower triangle, (i = $i, j = $j)"))
    elseif (k > K) | (k <= 0)
        throw(ArgumentError("Must access valid basis function coeff, (k = $k, K = $K)"))
    elseif c2 > length(blkdims) | (c2 <= 0)
        throw(ArgumentError("Must access valid block, (c = $c, C+1 = $(length(L.blockdims))"))
    elseif (j > blkdims[c2]) | (i > blkdims[c2])
        throw(ArgumentError("Must access valid matrix section, , (i = $i, j = $j, rc = $(L.blockdims[c2])))"))
    end
    # First to correct bs func, then to correct chan, then to col, then to row
    totLsize =  sum([s*(s+1) ÷ 2 for s in blkdims])# number of coefs on a given bs function
    prebasisoffset = (k-1) * totLsize
    preblksizes = blkdims[1:(c2-1)]
    preblkoffset = sum([(s*(s+1) ÷ 2) for s in preblksizes]) # == 0 if c==0
    coloffset =  (j-1) * blkdims[c2] - ((j-2)*(j-1) ÷ 2) # rc + (rc-1) + (rc -j+1)
    rowoffset = (i-j+1)
    return prebasisoffset + preblkoffset + coloffset + rowoffset
end

#get (row, col) position of cth channel column j row i in LT from_LT
function getelempos(blkdims, c, j, i)
    c2 = c+1
    preblksizes = blkdims[1:(c2-1)]
    preblkoffset = sum(preblksizes)
    return (preblkoffset + i, preblkoffset + j)
end

getelem(L :: MFACholeskyBSplineFunc, k, c, j, i) = L.coeffs[getelemidx(L.blockdims, length(L.bs), k, c, j, i)]
function setelem(s, L :: MFACholeskyBSplineFunc, k, c, j, i)
    L.coeffs[getelemidx(L.blockdims, length(L.bs), k, c, j, i)] = s
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
        riem_int += vl * pi / (length(ws)) # assume ws equally spaced
    end
    R = cholesky(riem_int).L
    # set C -> C*R, L(w) -> R^-1 * L(w) 
    C[:, :] = C * R 
    Lts = unpackLTsequence(L.coeffs, L.blockdims, length(L.bs))
    Ltsnew = [inv(R)  * Lts[k] for k in 1:length(L.bs)]
    packLTsequence!(L.coeffs, Ltsnew, L.blockdims)
end


function unitloadings_invariant!(C :: Matrix{<:Real}, L :: MFACholeskyBSplineFunc)
    cnorms = norm.(eachcol(C))
    normalize!.(eachcol(C))
    Lts = unpackLTsequence(L.coeffs, L.blockdims, length(L.bs))
    Ltsnew = [diagm(cnorms) * Lts[k] for k in 1:length(L.bs)]
    packLTsequence!(L.coeffs, Ltsnew, L.blockdims)
end


# Enforce that the diagonal of the cholesky factor be real and positive.
function realdiag_invariant!(L :: MFACholeskyBSplineFunc)
    K = length(L.bs)
    C  = length(L.blockdims)
    for k in 1:K
        for c2 in 1:C
            for j in 1:L.blockdims[c2]
                idx = getelemidx(L.blockdims,length(L.bs), k, c2-1, j, j)
                v = L.coeffs[idx]
                realrotfactor = norm(v) / v#exp(-1*im*angle(v))
                for i in j:L.blockdims[c2]
                    idx2 = getelemidx(L.blockdims,length(L.bs), k, c2-1, j, i)
                    L.coeffs[idx2] *= realrotfactor
                end
            end
        end
    end
end

function realdiag_invariant2!(L)
    K = length(L.bs)
    C = length(L.blockdims)
    for k in 1:K
        for c2 in 1:C
            for j in 1:L.blockdims[c2]
                idx = getelemidx(L.blockdims, K, k, c2-1, j, j)
                v = L.coeffs[idx]
                setelem(abs(v), L, k, c2-1, j, j)
            end
        end
    end
end

# take real part of all diags, enforce that diag(L(0)) > 0 
function realdiag_invariant3!(L)
    K = length(L.bs)
    C = length(L.blockdims)
    for k in 1:K
        for c2 in 1:C
            for j in 1:L.blockdims[c2]
                idx = getelemidx(L.blockdims, K, k, c2-1, j, j)
                v = L.coeffs[idx]
                setelem(real(v), L, k, c2-1, j, j)
            end
        end
    end
end


# Rotate a tall nxm matrix so that top mxm block is lower-triangular with positive diagonals.
function lowertri_invariant!(A :: Matrix{<:Real})
    n, m = size(A)
    if m == 0 || isnothing(m)
        return
    end
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
    unblockdiag!(Bcsstore, C[:, (size(Astore)[2]+1):end], rowblksizes, colblksizes)
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