using LinearAlgebra
using Distributions
using Statistics
using BSplines
using OffsetArrays


struct Lspec
    dim
    splines
end



function val(L, w)
    out = zeros(L.dim, L.dim)
    for i in 1:L.dim
        for j in i:L.dim
            out[i,j] = L.splines[i, j](w)
        end
    end
    return out
end

basis = BSplineBasis(2, pi*(-1:0.5:1))

a1 = abs.(randn(length(basis)))
a2 = ones(length(basis))
a3 = zeros(length(basis))
spl = [Spline(basis, a1) Spline(basis, a3) 
       Spline(basis, a3) Spline(basis, a2)
]

L = Lspec(2, spl)

function sample_z(A, phi, L, T)
    N, r = size(A)
    ws = pi * -1.0:(2/T):1.0
    Ds = [val(L, w) * val(L, w)' for w in ws]
    vars = [(A*D*A' + phi) for D in Ds] 
    xs = [sqrt(var) * randn(N) for var in vars]
    return xs
end



function to_L(alpha, basis, dim)
    spls = []
    tl = 0
    for i in 1:dim
        for j in 1:dim
            if j >  i
                spl = Spline(basis, repeat([0.0], length(basis)))
            else
                spl = Spline(basis, alpha[(tl+1):(tl+length(basis))])
                tl += length(basis)
            end
            push!(spls, spl)
        end
    end
    Lout = Lspec(dim, reshape(spls, (dim, dim)))
end

function to_alpha(L)
    basis = L.splines[1,1].basis
    lb = length(basis)
    alpha = Float64[]
    for i in 1:L.dim
        for j in i:L.dim
            alpha = vcat(alpha, L.splines[i,j].coeffs)
        end
    end
    return alpha
end

function fit_L(zs, A, phi, alpha_init, basis; tol=1e-3, maxiter=1000)
    N, r = size(A)
    zw = sqrt(inv(phi)) .* zs
    alpha = alpha_init
    alphaold = deepcopy(alpha)  
    iter = 0
    AtA = A' * A
    while norm(alpha -alphaold ) > tol | (iter == 0)
        iter +=1
        if iter >= maxiter
            break
        end
        alphaold = deepcopy(alpha)
        Xi = I(r) 
    end
end