using LinearAlgebra
using Distributions
using Statistics
using BenchmarkTools
using Random
using BSplines
using OffsetArrays
using IterativeSolvers


function lowertri_rot(A)
    n, m = size(A)
    if n < m
        error("A not tall.")
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
m = 2
T = 1200
Ds = gen_Ds(m, T)  #repeat([I(2)], T)
Ds1 = gen_Ds(m, T)#repeat([Ds[1]], T)
Dbar = mean(Ds1)
Ds1 =  [inv(sqrt(Dbar)).*D.*inv(sqrt(Dbar)) for D in Ds1]
A0 = lowertri_rot(randn(n, m))
Phi = diagm(1 .+ rand(n))
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
        A = real((L*sqrt(inv(H))))
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

function Gvec(sz, Gts, ws, basis)
    out = zeros(ComplexF64, sz)
    K = length(basis)
    r = size(Gts[1])[1]
    lt = 0
    for j in 1:r
        for i in j:r
            for k in 1:K
                lt+=1
                out[lt] = sum([basis[k](ws[t])*conj(Gts[t][i, j]) for t in 1:length(ws)])
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
    tot = sum([2*tr(Gts[t]*val(L, ws[t])) for t in 1:length(ws)])
    return real(tot)
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

function fit_L(xs, A, phi, alpha_init, bs, ws; tol=1e-3, maxiter=1000)
    N, r = size(A)
    #xs = sqrt(inv(phi)) .* xs
    #A =  sqrt(inv(phi)) * A
    alpha = alpha_init
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
        Hts = [Rts[i] + Rts[i]*val(Lold, ws[i])'*A'*xs[i]*xs[i]'*A*val(Lold, ws[i])*Rts[i] for i in 1:length(Rts)]
        Gts = [A' * xs[i] * xs[i]'*A*val(Lold, ws[i])*Rts[i] for i in 1:length(Rts)]
        b = Gvec(length(alpha), Gts, ws, bs)
        Q = Hermitian(Dmat(Hts, AtA, ws, bs))
        alpha = (Q \ b)
        L = Lspec(r, bs, normalize(alpha, r, length(bs)))
        Ds = get_Ds(L, ws)
        obj = objective(xs, Ds, A, phi)
        nm = norm(alpha - alphaold)
        println("Iter: $iter --  Obj: $obj, Norm: $nm")
    end
    return alpha
end



#fit_MM_phi(xs, Ds1, A0, Diagonal(I(n)))

#out2= fit_MM(xs, Ds1, Ainit; maxiter=5000)




# test L

ws = pi * (-1.0:0.2:1.0)
bs = BSplineBasis(2, pi*(-1:0.5:1))

a1 = abs.(randn(length(bs)))
a2 = ones(length(bs))
a3 = zeros(length(bs))
spl = [Spline(basis, a1) Spline(basis, a3) 
       Spline(basis, a3) Spline(basis, a2)
]
alpha = vcat(a1, a3, a2)
L = Lspec(2, bs,alpha)

n = 10
r = 2
T = 500

rng = MersenneTwister(1232)
Ds = get_Ds(L, ws)
A = lowertri_rot(randn(n, r))
xs = sample_data(A, Ds, I(n), rng)
alpha_init = alpha


out = fit_L(xs, A, I(n), alpha_init, bs, ws)