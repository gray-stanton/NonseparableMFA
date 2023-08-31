using LinearAlgebra
using Distributions
using Statistics
#using BenchmarkTools
using Random
using BSplines
using OffsetArrays
using IterativeSolvers


function lowertri_rot(A)
    n, m = size(A)
    if n < m
        throw(ArgumentError("A not tall (n=$n, m=$m)"))
    end
    A1 = A[1:m, 1:m]
    A1LQ = lq(A1)
    # Rotate
    A = A * A1LQ.Q'
    # Assert sign orientation
    for i in 1:m
        if sign(A[i, i]) == -1
            A[:,i] = -1*A[:, i]
        end
    end
    return A
end


function gen_Ds(m, nds)
    Ds = [Hermitian(zeros(ComplexF64, m, m)) for _ in 1:nds]
    for i in 1:nds
        Z = randn(m, m) + im*randn(m, m)
        Ds[i] = Hermitian(Z*Z')
    end
    return Ds
end

function sample_data(A, Ds, Phi, rng)
    n, m = size(A)
    T = length(Ds)
    xs = [zeros(ComplexF64, n) for _ in 1:T]
    for i in 1:T
        xs[i] = sqrt(A*Ds[i]*A'+Phi) * randn(rng, ComplexF64, n)
    end
    return xs
end


rng = MersenneTwister(1234)
n = 10
m = 3
T = 600
Ds = gen_Ds(m, T)  #repeat([I(2)], T)
Ds1 = gen_Ds(m, T) 
Dbar = mean(Ds1)
Ds1 =  [inv(sqrt(Dbar)).*D.*inv(sqrt(Dbar)) for D in Ds1]
A0 = lowertri_rot(randn(n, m))
Phi = I#diagm(1 .+ rand(n))
xs = sample_data(A0, Ds1,Phi, rng)
Ainit = lowertri_rot(randn(n, m))
X = hcat(xs...)
avg_var = A0*A0'+Phi
sample_avg_var = 1/T * X * X'
sample_avg_psvar = 1/T * X * transpose(X)


function objective(xs, Ds, A, Phi)
    obj = 0.0
    T = length(xs)
    for t in 1:T
        obj += 1/T * log(det(A*Ds[t]*A'+Phi)) # (I + Ds[t]*A'*A)
        obj += 1/T*(xs[t]'* inv(Phi)* xs[t] - xs[t]'*inv(Phi)*A*Ds[t]*inv(Ds[t] + Ds[t]*A'* inv(Phi) * A*Ds[t])*Ds[t]*A'*inv(Phi)*xs[t])
    end
    return obj
end

function objective2(xs, Ds, A)
    obj = 0.0
    T = length(xs)
    for t in 1:T
        obj += 1/T * log(det(A*Ds[t]*A'+I))
        obj += 1/T*xs[t]' *inv(A*Ds[t]*A'+I)*xs[t]
    end
    return obj
end


function fit_MM(xs, Ds, Ainit; eps = 1e-4, maxiter=1000)
    n = length(xs[1])
    m = size(Ds[1])[1]
    T = length(Ds)
    A = deepcopy(Ainit)
    Aold = deepcopy(A)
    iter = 0
    while (norm(A - Aold) > eps) | (iter == 0)
        iter += 1
        if iter >= maxiter
            break
        end
        Aold = deepcopy(A)
        AtAold = Symmetric(Aold'*Aold)
        H = Hermitian(zeros(Float64, m, m))
        L = zeros(Float64, n, m)
        for t in 1:T
            #Rt = Hermitian(Ds[t] * inv(Hermitian(Ds[t]+Ds[t]*AtAold*Ds[t]))*Ds[t])
            Rt = inv(inv(Ds[t]) + AtAold)
            Ht = Hermitian(Rt + Rt*Aold'*xs[t]*xs[t]'*Aold*Rt)
            Lt = xs[t]*xs[t]'*Aold*Rt
            H += 1/T * Ht
            L += 1/T * Lt
        end
        A = real((L*(inv(H))))
        obj = objective(xs, Ds, A, I)
        nm = norm(A - Aold)
        println("Iter: $iter --  Obj: $obj, Norm: $nm")
    end
    return lowertri_rot(A)
end


function fit_MM_phi(xs, Ds, A, Phiinit; eps = 1e-4, maxiter=1000)
    n = length(xs[1])
    m = size(Ds[1])[1]
    T = length(Ds)
    Phi = deepcopy(Phiinit)
    Phiold = deepcopy(Phi)
    iter = 0
    Dsinvs = [inv(D) for D in Ds]
    while (norm(Phi - Phiold) > eps) | (iter == 0)
        iter += 1
        if iter >= maxiter
            break
        end
        Phiold = deepcopy(Phi)
        Rts = [inv(Dsi + A'*inv(Phiold)*A) for Dsi in Dsinvs]
        ws = [A*Rts[i] * A'* inv(Phiold)*xs[i] for i in 1:length(xs)]
        Vs = [(xs[i] - ws[i])*(xs[i] - ws[i])' + A*Rts[i]*A' for i in 1:length(xs)]
        V = mean(Vs)
        Phi = Diagonal(real.(diag(V)))
        obj = objective(xs, Ds, A, Phi)
        nm = norm(Phi - Phiold)
        println("Iter: $iter --  Obj: $obj, Norm: $nm")
    end
    return Phi
end






struct Lspec
    dim
    basis
    coefvec
end



function val(L, w)
    out = zeros(eltype(L.coefvec), L.dim, L.dim)
    lt = 0
    K = length(L.basis)
    lt = 0
    for j in 1:L.dim
        for i in j:L.dim
            lt +=1
            vs = bsplines(L.basis,w)
            if isnothing(vs)
                continue
            end
            for (k, v) in pairs(vs)
                idx = K*(lt-1) + k
                out[i,j] += v * L.coefvec[idx]
            end
        end
    end
    return out
end


function get_Ds(L, ws)
    Ds = [val(L, w) * val(L, w)' for w in ws]
    return Ds
end



# function to_L(alpha, basis, dim)
#     spls = []
#     tl = 0
#     for i in 1:dim
#         for j in 1:dim
#             if j >  i
#                 spl = Spline(basis, repeat([0.0], length(basis)))
#             else
#                 spl = Spline(basis, alpha[(tl+1):(tl+length(basis))])
#                 tl += length(basis)
#             end
#             push!(spls, spl)
#         end
#     end
#     Lout = Lspec(dim, reshape(spls, (dim, dim)))
# end

# function to_alpha(L)
#     basis = L.splines[1,1].basis
#     lb = length(basis)
#     alpha = Float64[]
#     for i in 1:L.dim
#         for j in i:L.dim
#             alpha = vcat(alpha, L.splines[i,j].coeffs)
#         end
#     end
#     return alpha
# end


function Dmult(alpha, Hts, AtA, ws, basis)
    r = size(AtA)[1]
    T = length(Hts)
    K = length(basis)
    out = zeros(length(alpha))
    lt = 0
    for j1 in 1:r
        for i1 in j1:r
            for k1 in 1:K
                lt += 1
                acc = 0
                lr = 0
                for j2 in 1:r
                    for i2 in j2:r
                        for k2 in 1:K
                            lr +=1
                            acc += sum([Hts[t][j2,j1] * basis[k2](ws[t]) * basis[k1](ws[t]) * AtA[i1, i2] * alpha[lr] for t in 1:T])
                        end
                    end
                end
                out[lt] = acc
            end
        end
    end
    return out
end

function Dmat(Hts, AtA, ws, basis)
    r = size(AtA)[1]
    T = length(Hts)
    K = length(basis)
    sz = Int(K*r*(r+1)/2)
    out = zeros(ComplexF64, sz, sz)
    lt = 0
    for j1 in 1:r
        for i1 in j1:r
            for k1 in 1:K
                lt+=1
                lr = 0
                for j2 in 1:r
                    for i2 in j2:r
                        for k2 in 1:K
                            lr+=1
                            out[lt, lr] = sum([Hts[t][j2,j1] * basis[k2](ws[t]) * basis[k1](ws[t]) * AtA[i1, i2] for t in 1:T])
                        end
                    end
                end
            end
        end
    end
    return out
end

function Gvec(sz, Gts, ws, bs)
    out = zeros(ComplexF64, sz)
    K = length(bs)
    r = size(Gts[1])[1]
    lt = 0
    for j in 1:r
        for i in j:r
            for k in 1:K
                lt+=1
                out[lt] = sum([bs[k](ws[t])*(Gts[t][i,j]) for t in 1:length(ws)])
            end
        end
    end
    return out
end


function Dform_eval(alpha, Hts, AtA, ws, basis)
    r = size(AtA)[1]
    L = Lspec(r, basis, alpha)
    tot = 0
    for t in 1:length(ws)
        tot += tr(Hts[t]*val(L, ws[t])' * AtA * val(L, ws[t]))
    end
    return tot
end

function Gpart_eval(alpha, Gts, bs, ws)
    r = size(Gts[1])[1]
    L = Lspec(r, bs,  alpha)
    tot = sum([tr(Gts[t]*val(L, ws[t])') for t in 1:length(ws)])
    return 2*real(tot)
end

function normalize(alpha, r, K)
    lt = 0
    out = deepcopy(alpha)
    for j in 1:r
        for i in j:r
            for k in 1:K
                lt+=1
                if i == j
                    out[lt] = norm(alpha[lt])
                end
            end
        end
    end
    return out
end

function fix_invariant(A, alpha, K, ws, bs)
    r = size(A)[2]
    L = Lspec(r, bs, alpha)
    Ds = get_Ds(L, ws)
    sz = Int(r*(r+1)/2)
    Dmean = 2*real(mean(Ds))
    R = cholesky(Dmean).U'
    Anew = A * R
    Rinv = inv(R)
    apack = unpack_alpha_by_K(alpha, K, r)
    Lts = [to_LT(apack[i, :], r) for i in 1:K]
    Lts = [Rinv * L for L in Lts]
    apackout = zeros(eltype(alpha), K, sz)
    for k in 1:K
        apackout[k, :] = from_LT(Lts[k], r)
    end
    alphaout = pack_alpha_by_K(apackout, K, r)
    return (Anew, alphaout)
end

function to_LT(v, r)
    out = zeros(eltype(v), r, r)
    lt = 0
    for j in 1:r
        for i in j:r
            lt+=1
            out[i, j] = v[lt]
        end
    end
    return LowerTriangular(out)
end

function from_LT(L, r)
    sz = Int(r * (r+1)/2)
    out = zeros(eltype(L), sz)
    lt = 0
    for j in 1:r
        for i in j:r
            lt+=1
            out[lt] = L[i, j]
        end
    end
    return out
end

function pack_alpha_by_K(alphas,K, r)
    sz = Int(r*(r+1)/2*K)
    alphaout = zeros(ComplexF64, sz)
    lt = 0
    for j in 1:r
        for i in j:r
            for k in 1:K
                lt+=1
                alphaout[lt] = alphas[k, (j-1)*r+(i-j+1)]
            end
        end
    end
    return alphaout
end

function unpack_alpha_by_K(alpha, K, r)
    sz = Int(length(alpha)/K)
    out = zeros(ComplexF64, K, sz)
    lt = 0
    for j in 1:r
        for i in j:r
            for k in 1:K
                lt+=1
                out[k, (j-1)*r+(i-j+1)] = alpha[lt]
            end
        end
    end
    return out
end




function fit_L(xs, A, phi, alpha_init, bs, ws; tol=1e-6, maxiter=1000)
    N, r = size(A)
    #xs = sqrt(inv(phi)) .* xs
    #A =  sqrt(inv(phi)) * A
    alpha = deepcopy(alpha_init)
    alphaold = deepcopy(alpha)
    iter = 0
    while (norm(alpha - alphaold ) > tol) | (iter == 0)
        iter +=1
        if iter >= maxiter
            break
        end
        AtA = A' * A
        alphaold = deepcopy(alpha)
        Lold = Lspec(r, bs, alphaold)
        Rts= [inv(I + val(Lold, w)' * AtA * val(Lold, w)) for w in ws]
        Hts = 1/length(ws) .* [Rts[i] + Rts[i]*val(Lold, ws[i])'*A'*xs[i]*xs[i]'*A*val(Lold, ws[i])*Rts[i] for i in 1:length(Rts)]
        Gts = 1/length(ws) .* [A' * xs[i] * xs[i]'*A*val(Lold, ws[i])*Rts[i] for i in 1:length(Rts)]
        b = Gvec(length(alpha), Gts, ws, bs)
        Q = Hermitian(Dmat(Hts, AtA, ws, bs))
        diff = norm(alpha'*Q*alpha - 2*real(alpha'*b) - Dform_eval(alpha, Hts, AtA, ws, bs) + Gpart_eval(alpha, Gts, bs, ws))
        alpha = (Q \ b)
        #A, alpha = fix_invariant(A, alpha, length(bs), ws, bs)
        #alpha = normalize(alpha, r, length(bs))
        L = Lspec(r, bs, alpha)
        Ds = get_Ds(L, ws)
        obj = objective(xs, Ds, A, phi)
        nm = norm(alpha - alphaold)
        println("Iter: $iter --  Obj: $obj, Norm: $nm, Diff: $diff")
    end
    return alpha
end




#fit_MM_phi(xs, Ds1, A0, Diagonal(I(n)))

out = fit_MM(xs, Ds1, Ainit; maxiter=5000)




# test L
n = 5
r = 2
T = 1000


ws = pi * (-1.0:1/T:1.0)
bs = BSplineBasis(2, pi*(-1:0.5:1))

a1 = abs.(randn(length(bs)))
a2 = ones(length(bs))
a3 = 0.02*randn(ComplexF64, length(bs))

alpha0 = vcat(a1, a3, a2)
L0 = Lspec(2, bs,alpha0)



Dsold = get_Ds(L0, ws)
rng = MersenneTwister(1232)
A = lowertri_rot(randn(n, r))
Anew, alphanew = fix_invariant(A, alpha0, length(bs), ws, bs)
Lnew = Lspec(2, bs, alphanew)
Dsnew = get_Ds(Lnew, ws)


xs = sample_data(Anew, Dsnew, I(n), rng)
alpha_init = vcat(ones(ComplexF64, length(bs)), zeros(ComplexF64, length(bs)), ones(ComplexF64, length(bs)))


#out = fit_L(xs, deepcopy(Anew), I(n), alpha_init, bs, ws; maxiter=500)


# Lout = Lspec(r, bs, out)
# Dsout = get_Ds(Lout ,ws)
# objout = objective(xs, Dsout, Anew, I(n))
# objtrue = objective(xs, Dsnew, Anew, I(n))

# Aout, alphaout = fix_invariant(Anew, out, length(bs), ws, bs)

# Lout2 = Lspec(r, bs, alphaout)
# Dsout2 = get_Ds(Lout2, ws)
# objout2 = objective(xs, Dsout2, Aout, I(n))

# for i in 1:length(alpha0)
#     println("Estim: $(alphaout[i]), Act: $(alphanew[i])")
# end

# spec_dens_out = [Anew*Dsout[i]*Anew'+I(n) for i in 1:length(Dsout)]
# spec_dens_act = [Anew*Dsnew[i]*Anew'+I(n) for i in 1:length(Dsout)]
# spec_dens_out2 = [Aout*Dsout2[i]*Aout'+I(n) for i in 1:length(Dsout)]


# total_err = mean([norm(spec_dens_act[i] - spec_dens_out[i])/norm(spec_dens_act[i]) for i in 1:length(spec_dens_act)])


# sp11act = [real(spec_dens_act[i][1,1]) for i in 1:length(ws)]
# sp11out = [real(spec_dens_out[i][1,1]) for i in 1:length(ws)]

# sp22act = [real(spec_dens_act[i][3,3]) for i in 1:length(ws)]
# sp22out = [real(spec_dens_out[i][3,3]) for i in 1:length(ws)]

# sp12act = [real(spec_dens_act[i][1,2]) for i in 1:length(ws)]
# sp12out = [real(spec_dens_out[i][1,2]) for i in 1:length(ws)]


# sp55act = [real(spec_dens_act[i][5,5]) for i in 1:length(ws)]
# sp55out = [real(spec_dens_out[i][5,5]) for i in 1:length(ws)]

# plot(ws, hcat(sp11act, sp11out), ylims=(0.8*minimum(sp11act), 1.2*maximum(sp11act)))
# plot(ws, hcat(sp22act, sp22out),  ylims=(0.8*minimum(sp22act), 1.2*maximum(sp22act)))
# plot(ws, hcat(sp12act, sp12out))

# plot(ws, hcat(sp55act, sp55out), ylims=(0.8*minimum(sp55act), 1.2*maximum(sp55act)))
