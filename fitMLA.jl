
using LinearAlgebra
using Distributions
using FFTW


function VARMA_sim(nobs, coefs, init_vals, innov_sample_dist)
    lags=length(coefs)
    pops = size(coefs[1])[1]
    result = zeros(eltype(init_vals), pops, nobs)
    result[:, 1:lags] = init_vals[:, 1:lags]
    for t in (lags+1):nobs
        result[:, t] = rand(innov_sample_dist)
        for tau in 1:lags
            result[:, t] += coefs[tau] * result[:, t - tau] 
        end
    end
    return result 
end




function MFA_sim(nobs, A, Bs, Phi, latent_facs, innovs)
    pops = size(A)[1]
    r0 = size(A)[2]
    Ns = [size(Bc)[1] for Bc in Bs]
    rs = [size(Bc)[2] for Bc in Bs]
    
    X = zeros(pops, nobs)
    for t in 1:nobs
       X[:, t] = Phi * innovs[:, t]
       X[:, t] += A * latent_facs[1:r0, t]
       for c in 1:length(Bs)
            X[(sum(Ns[1:(c-1)])+1):sum(Ns[1:c]), t] += Bs[c] * latent_facs[(r0+sum(rs[1:(c-1)])+1):(r0+sum(rs[1:c])), t]
       end
    end
    return X
end

r0=2
rs=[1,1]
Ns=[10, 10]
N = sum(Ns)
nobs=1000
A = randn(N, r0)
Bs = [randn(n, r) for (n, r) in zip(Ns, rs)]
arcoef = 0.0
sig = sqrt(1-arcoef^2)
latent_facs = VARMA_sim(nobs, [diagm(repeat([arcoef], r0+sum(rs)))], zeros(r0+sum(rs), 1), MvNormal(r0+sum(rs), sig))
innovs = randn(N, nobs)

X= MFA_sim(nobs, A, Bs, diagm(repeat([1.0], N)), latent_facs, innovs)
Z = 1/sqrt(2*pi*nobs) * fft(X, 2)

Φ = diagm(repeat([sig^2], N))

specdens = [Hermitian(zeros(ComplexF64, r0+sum(rs), r0+sum(rs))) for _ in 1:nobs]
for r in 1:(r0+sum(rs))
    for w in 1:size(Z)[2]
        specdens[w][r,r] = 1.0
    end
end

function stepAB(Z, Λ :: Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}}, Φold, Aold, Bsold)
    N, T = size(Z)
    r = size(Aold)[2]
    Ts = [zeros(ComplexF64, N, r) for _ in 1:T]
    Us = [Hermitian(zeros(ComplexF64, r, r)) for _ in 1:T]

    Zwhite = sqrt(inv(Φold)) * Z
    Cwhite = sqrt(inv(Φold)) * toC(Aold, Bsold)

    Jwhite = [Zwhite[:, w] * Zwhite[:, w]' for w in 1:T]
    CtC = Cwhite' * Cwhite
    Λinv = [inv(Λ[w]) for w in 1:T]
    Gs = [inv(Λinv[w] + CtC) for w in 1:T]

    Ts = [Jwhite[w] * Cwhite * Gs[w] for w in 1:T]
    Us = [Gs[w]  + Gs[w] * Cwhite' * Jwhite[w] * Cwhite * Gs[w] for w in 1:T]

    Tmean = mean(Ts)
    Umean = mean(Us)
    
    Hnew = real(Tmean * sqrt(inv(Umean)))
    Anew, Bsnew = toABs(Hnew, size(A)[2], [size(Bc)[1] for Bc in Bs], [size(Bc)[2] for Bc in Bs])
    return (Anew, Bsnew)
end

function stepPhi(Z, Λ :: Vector{Hermitian{ComplexF64, Matrix{ComplexF64}}}, Φold, Aold, Bsold)
    N, T = size(Z)
    r = size(Aold)[2]
    Cold = toC(Aold, Bsold)
    CtC = Cold' * inv(Φold) * Cold

    Λinv = [inv(Λ[w]) for w in 1:T]
    Gs = [inv(Λinv[w] + CtC) for w in 1:T]

    w = [Cold * Gs[t] * Cold' * inv(Φold) * Z[:, t] for t in 1:T]
    Vs = [(Z[:, t] - w[t])*(Z[:, t] - w[t])' + Cold*Gs[t]*Cold' for t in 1:T]

    V = mean(Vs)
    return(Diagonal(V))
end



function toABs(C, r0, Ncs, rcs)
   A = C[:, 1:r0]
   Bs = [zeros(Nc, rc) for (Nc, rc) in zip(Ncs, rcs)]
   for c in 1:length(Bs)
        init_row = c == 1 ? 1 : sum(Ncs[c-1])+1
        init_col = c == 1 ? (r0 + 1) : sum(rcs[c-1]) + r0 + 1
        Bs[c] = C[init_row:(init_row+Ncs[c]-1), init_col:(init_col+rcs[c]-1)]
   end
   return (A, Bs)
end

function toC(A, Bs)
    N, r0 = size(A)
    Ncs = [size(Bc)[1] for Bc in Bs]
    rcs = [size(Bc)[2] for Bc in Bs]
    C = zeros(Float64, N, r0 + sum(rcs))
    C[:, 1:r0] = A
    for c in 1:length(Ncs)
        init_row = c == 1 ? 1 : sum(Ncs[c-1])+1
        init_col = c == 1 ? (r0 + 1) : sum(rcs[c-1]) + r0 + 1
        C[init_row:(init_row+Ncs[c]-1), init_col:(init_col+rcs[c]-1)] = Bs[c]
    end
    return C
end


function objective(Z, specdens, A, Bs, Φ)
    obj = 0.0
    C = toC(A, Bs)
    for t in 1:size(Z)[2]
        obj += log(det(C*specdens[t]*C' + Φ))
        obj += Z[:,t]'*inv(C*specdens[t]*C' + Φ)*Z[:,t]
    end
    return obj/size(X)[2]
end 

Anew = A
Bsnew = Bs
Φnew = Φ
for _ in 1:100
    Anew, Bsnew = stepAB(X, specdens, Φnew, Anew, Bsnew)
    #Φnew = stepPhi(X, specdens, Φnew, Anew, Bsnew)
end

objective(X, specdens, A, Bs, Φ)
objective(X, specdens, Anew, Bsnew, Φnew)
