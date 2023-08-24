using LinearAlgebra
using BSplines
using Distributions
using Random
include("./MFADependentUtils.jl")

function sample_zs(rng, ws, model :: MFADependent)
    N = model.cs.N
    zs = [zeros(eltype(model.L.coeffs), N) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        V = specdens(model, w)
        zs[t] = sqrt(V) * randn(rng, eltype(model.L.coeffs), N)
    end
    return zs
end



function sample_VAR1(rng, T, coeffmat, innovsd; burnin=100)
    n = size(coeffmat)[1]
    zs = [zeros(eltype(coeffmat), n) for _ in 1:T]
    z = zeros(eltype(coeffmat), n)
    for t in (-burnin+1):T
        z = coeffmat * z + innovsd * randn(rng, eltype(coeffmat), n)
        if t > 0
            zs[t] = z
        end
    end
    return zs
end

AR1_specdens(w, p, σ) = σ^2 / (2*pi * (1-2*p*cos(w)+p^2))


function VAR1_covariance(h, coeffmat, innovsd)
    if h == 0
        return AR1_variance(coeffmat, innovsd) 
    elseif h > 0
        return coeffmat^(abs(h))*AR1_variance(coeffmat, innovsd)
    else
        return (coeffmat^(abs(h))*AR1_variance(coeffmat, innovsd))'
    end
end

# B+D p405
function VAR1_specdens(w, coeffmat, innovsd; hlim=100)
    spec = AR1_variance(coeffmat, innovsd)
    for h in 1:hlim
        spec += exp(-im*w*h) * VAR1_covariance(h, coeffmat, innovsd) 
        spec += exp(im*w*h)*VAR1_covariance(-h,coeffmat, innovsd)
    end
    return spec/(2*pi)
end

function get_co(w, coeffmat, innovsd; hlim=100)
    spec = AR1_variance(coeffmat, innovsd)
    for h in 1:hlim
        spec += exp(-im*w*h) * VAR1_covariance(h, coeffmat, innovsd) 
        spec += exp(im*w*h)*VAR1_covariance(-h,coeffmat, innovsd)
    end
    return spec/(2*pi)
end

function slow_fourier_transform(xs)
    T = size(xs)[2]
    J = zeros(ComplexF64, size(xs)[1], Int(ceil(T/2) + 1))
    for j in 0:Int(ceil(T/2))
        wj = 2*pi*j/(T)
        J[:, (j+1)] = 1/sqrt(T) * sum([xs[:, n:n] * exp(-im*n*wj) for n in 1:T])
    end
    return J
end

function optimal_Bspline_coeffs(specdens_func, ws, bs)
    # set all coeffs to one first
    true_specdens = [specdens_func(w) for w in ws]
    true_specdens_mat = hcat([halfvec(D) for D in true_specdens]...)
    bsvals = hcat([s for w in ws]...)
    #TODO FINISH
end

function best_fit_chol(specdens_func, bs, bdims, ws)
    Lout = MFACholeskyBSplineFunc(bs, bdims)
    W = length(Lout.coeffs)
    L = W ÷ length(bs)
    M = length(bs)
    B = zeros(M, length(ws))
    V = zeros(L, length(ws))
    for (i,w) in enumerate(ws)
        spd = specdens_func(w)
        Lexact = cholesky(spd).L
        bds = unblockdiag(Lexact, bdims, bdims)
        vecs = [halfvec(bd) for bd in bds]
        v = vcat(vecs...)
        bsplinevals = bsplines(bs, w)
        bsvals = zeros(M)
        if !isnothing(bsplinevals)
            for (k, v) in pairs(bsplinevals)
                bsvals[k] = v
            end
        end
        B[:, i] = bsvals
        V[:, i] = v
    end
    A = V / B
    bestcoeffs = reshape(A, W)
    Lout.coeffs[:] = bestcoeffs
    identityintegral_invariant!(zeros(W, sum(bdims)), Lout, ws)
    return Lout
end



function invert_specdens(h, ws, specdens_func)
    tot = zeros(Float64, size(specdens_func(0)))
    for w in ws
        tot += 2*real(exp(im*w*h)*specdens_func(w)) * pi/(length(ws))
    end
    return tot / (2*pi)
end

function sample_xs(rng, facs, C, P)
    T = length(facs)
    N = size(C)[1]
    xs = [zeros(eltype(C), N) for _ in 1:T]
    for t in 1:T
        xs[t] = C*facs[t] + sqrt(P) * randn(rng, eltype(C), N)
    end
    return xs
end

function AR1_variance(coeffmat, innovsd)
    vvar = inv(I - kron(coeffmat, coeffmat)) * vec(innovsd*innovsd)
    return reshape(vvar, size(coeffmat))
end

ident_AR1_innovsd(coeffmat) = sqrt(Symmetric(abs.(I - coeffmat * coeffmat')))
