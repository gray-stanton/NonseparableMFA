

function fit_indep(xs, model; ftol=1e-8, xtol=1e-5, maxiter=30)
    T = length(xs)
    X = hcat(xs...)
    S = 1/T * X * X'
    ws = 2*pi*rfftfreq(T)
    Z = 1/sqrt(T)*rfft(X, 2)
    zs = [Z[:, t] for t in 1:(size(Z)[2])]
    trc = []
    obj = QGML_objective_full(zs, ws, model)
    oldobj = obj
    iter = 0
    while iter < maxiter
        oldobj = obj
        cs = model.cs
        model.A = Hstep(S, model.B, model.P, model.cs.N, model.cs.r0)
        Sblocks = unblockdiag(S, cs.Ncs, cs.Ncs)
        idx = 0
        for c in 1:cs.C
            Ac = model.A[(idx+1):(idx+cs.Ncs[c]), :]
            Pc = model.P[(idx+1):(idx+cs.Ncs[c]), (idx+1):(idx+cs.Ncs[c])]
            Bcnew = _Gstep_channel(Sblocks[c], Ac, Diagonal(Pc), model.cs.rcs[c])
            model.Bcs[c] = Bcnew
            idx+=cs.Ncs[c]
        end
        blockdiag!(model.B, model.Bcs)
        model.P = Diagonal(Σstep(S, model.A, model.B))
        obj = indep_objective(S, model)
        model.C = hcat(model.A, model.B)
        println("Iter $iter Obj: $obj")
        push!(trc, deepcopy(model))
        iter += 1
    end
    return model, trc
end

function indep_objective(S, model)
    R = model.C*model.C' + model.P
    obj = log(det(R)) + tr(inv(R)*S)
    return obj
end

function Hstep(S, G, Σ, L, p)
    if p == 0
        # if no channel-shared factors, return empty array of correct dims.
        Hnew= zeros(L, 0)
        return Hnew
    end
    E = G * transpose(G) + Σ # eqn 24
    Eisqrt = zeros(size(E))
    Esqrt = sqrt(E)
    try
        Eisqrt = inv(Esqrt) # inefficient
    catch e
        print(e)
        throw(e)
        #Eisqrt = inv(Esqrt + diagm(repeat([0.001], size(E)[1]))) # underflow problems
    end
    Swhite = Eisqrt * Symmetric(S) * Eisqrt # eqn 25
    Swhite_eigen=SVD{Float64, Float64, Matrix{Float64}}
    try
        Swhite_eigen=svd(Swhite,alg=LinearAlgebra.DivideAndConquer())
    catch e
        print(e)
        Swhite_eigen=svd(Swhite,alg=LinearAlgebra.QRIteration())
    end
    W = Swhite_eigen.U

    #TODO: Notation in paper is ambigious, double check defn of D.
    ds = max.(Swhite_eigen.S[1:p]  .- 1, zeros(p))
    Dsqrt = vcat(sqrt(diagm(ds)) , zeros((L-p, p)))

    B = Esqrt * W * Dsqrt # before eqn 11
    Q = LTOrthog(B, p)
    Hnew = real.(B  * Q)
    return Hnew
end


function _Gstep_channel(S, H, Σ, pj)
    Lj = size(H)[1]
    P = S - H * transpose(H) # eqn 44
    Σsqrt = sqrt(Σ)
    Σisqrt = zeros(size(Σ))
    try
        Σisqrt = inv(Σsqrt)
    catch e
        print(e)
        throw(e)
        #Σisqrt = inv(Σsqrt+ diagm(repeat([0.01], size(Σ)[1])))
    end

    Pwhite = Σisqrt * Symmetric(P) * Σisqrt
    Pwhite_eigen=SVD{Float64, Float64, Matrix{Float64}}
    try
        Pwhite_eigen=svd(Pwhite,alg=LinearAlgebra.DivideAndConquer())
    catch e
        print(e)
        Pwhite_eigen=svd(Pwhite,alg=LinearAlgebra.QRIteration())
    end
    W = Pwhite_eigen.U

    ds = max.(Pwhite_eigen.S[1:pj] .-1, zeros(pj))
    Dsqrt = vcat(sqrt(diagm(ds)) , zeros((Lj-pj, pj))) #TODO: Still questionable.

    B = Σsqrt * W * Dsqrt
    Q = LTOrthog(B, pj)
    Gnew  = real.(B * Q)
    return Gnew
end



function LTOrthog(B, blocksize)
    Blq = lq(B[1:blocksize, 1:blocksize])
    # Extract diagonal elems, sort into descending abs val, put back in diag mat.
    Lsigns = diagm(sign.(sort(diag(Blq.L), lt=(x, y) -> abs(y) < abs(x) )))
    Q = transpose(Blq.Q) * Lsigns  # eqn 11
    return Q
end


function Σstep(S, H, G; Pfloor = 0.001)
    Est = S - H * transpose(H) - G * transpose(G)
    Σnew = Diagonal(Est) # extract only the diagonal elems.
    Σnew = max.(Σnew, Pfloor)
    return Σnew
end