using LinearAlgebra
using BSplines
using Statistics
using OffsetArrays
using FFTW
using SparseArrays
include("./MFADependentUtils.jl")


# 1/T sum[logdet (specdens(A,B,P,α)(w)) + z(w)'* inv(specdens(A,B,P,α)(w))z(w)]
function QGML_objective_full(zs, ws, model :: MFADependent)
    tot = 0.0
    for (z, w) in zip(zs, ws)
        V = specdens(model, w)
        tot += log(det(V)) + z'*inv(V)*z
    end
    return real(tot)/length(zs)
end


function QGML_objective(zs, ws, model :: MFADependent)
    tot = 0.0
    Pinv = inv(model.P)
    C = model.C
    for (z, w) in zip(zs, ws)
        Lv = model.L(w)
        H = I + Lv'*C'*Pinv*C*Lv
        tot += log(det(H)) + log(det(model.P))
        tot += z'*Pinv*z - z'*Pinv*C*Lv*inv(H)*Lv'*C'*Pinv*z
    end
    return real(tot)/length(zs)
end


function QGML_objective_comp(zs, ws, model :: MFADependent)
    tot = 0.0
    Pinv = inv(model.P)
    C = model.Ccomp
    for (z, w) in zip(zs, ws)
        Lv = model.L(w)
        H = I + Lv'*C'*Pinv*C*Lv
        tot += log(det(H)) + log(det(model.P))
        tot += z'*Pinv*z - z'*Pinv*C*Lv*inv(H)*Lv'*C'*Pinv*z
    end
    return real(tot)/length(zs)
end


function AB_minorizing_obj_parts(zs, ws, Cnew, model_old)
    C, L, P= model_old.C, model_old.L, model_old.P
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    objs1 = [real(tr(Xis[k] * Cnew' * Pinv * Cnew)) for k in 1:length(ws)]
    objs2 = [real(tr(Xis[k] * Cnew' * Pinv * zs[k] * zs[k]' * Pinv * Cnew)) for k in 1:length(ws)]
    return (objs1 ./ T , objs2 ./ T, (objs1 + objs2)./ T)
end

function AB_minorizing_objective(zs, ws, Cnew, model_old)
    C, L, P= model_old.C, model_old.L, model_old.P
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    # Compute least-square mats
    Hs = [(Xis[t] + Xis[t]*Cw'*zws[t]*zws[t]'*Cw*Xis[t]) for t in 1:length(ws)]
    Us = [zs[t]*zws[t]'*Cw*Xis[t] for t in 1:length(ws)]
    H  = mean(Hs)
    U = mean(Us)
    #obj = tr(Psqinv * Cnew * H * Cnew' * Psqinv) - 2 * real(tr(Psqinv * U * Cnew' * Psqinv))
    obj =  norm(Psqinv * U * sqrt(inv(H)) - Psqinv * Cnew * sqrt(H))^2
    return real(obj)
end

function make_K(H, P, chanspec)
    chancount, N, Ncs, r0, rcs = chanspec.C, chanspec.N, chanspec.Ncs, chanspec.r0, chanspec.rcs
    rtot = r0 + sum(rcs)
    entry_count = N*r0 + sum([Nc * rc for (Nc, rc) in zip(Ncs, rcs)])
    K = zeros(eltype(P), 2*N*rtot, entry_count)
    return K
end

function make_K2(Hsq, Psq, chanspec)
    full_K = kron(conj.(Hsq), Psq)
    chancount, N, Ncs, r0, rcs = chanspec.C, chanspec.N, chanspec.Ncs, chanspec.r0, chanspec.rcs
    Afake = ones(eltype(Psq), N, r0)
    Bcsfake = [ones(eltype(Psq), Nc, rc) for (Nc, rc) in zip(Ncs, rcs)]
    Bfake = blockdiag(Bcsfake)
    Cfake = hcat(Afake, Bfake)
    indices = findnz(sparse(vec(Cfake)))[1]
    return full_K[:, indices]
end




function w_to_C(w, chanspec)
    K, N, Ncs, r0, rcs = chanspec.C, chanspec.N, chanspec.Ncs, chanspec.r0, chanspec.rcs
    w_a = w[1:(N*r0)]
    A = reshape(w_a, N, r0)
    Bcs = Matrix{eltype(w)}[]
    idx = (N*r0)
    for c in 1:K
        w_bc = w[(idx+1):(idx+Ncs[c]*rcs[c])]
        Bc = reshape(w_bc, Ncs[c], rcs[c])
        push!(Bcs, Bc)
        idx += Ncs[c]*rcs[c]
    end
    B = blockdiag(Bcs)
    C = hcat(A, B)
    return C
end

function C_to_w(C, chanspec)
    K, N, Ncs, r0, rcs = chanspec.C, chanspec.N, chanspec.Ncs, chanspec.r0, chanspec.rcs
    A, Bcs = extractABcs(C, Ncs, r0, rcs)
    w_a = reshape(A, N*r0)
    wbs = Vector{eltype(C)}[]
    for c in 1:K
        wb = reshape(Bcs[c], Ncs[c]*rcs[c])
        push!(wbs, wb)
    end
    w = vcat(w_a, wbs...)
    return w
end


function AB_minorizing_objective_comp(zs, ws, Cnew, model_old)
    C, L, P= model_old.Ccomp, model_old.L, model_old.P
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    # Compute least-square mats
    Hs = [(Xis[t] + Xis[t]*Cw'*zws[t]*zws[t]'*Cw*Xis[t]) for t in 1:length(ws)]
    Us = [zs[t]*zws[t]'*Cw*Xis[t] for t in 1:length(ws)]
    H  = mean(Hs)
    U = mean(Us)
    #obj = tr(Psqinv * Cnew * H * Cnew' * Psqinv) - 2 * real(tr(Psqinv * U * Cnew' * Psqinv))
    obj =  norm(Psqinv * U * sqrt(inv(H)) - Psqinv * Cnew * sqrt(H))^2
    return real(obj)
end


function QGML_objective_parts(zs, ws, model :: MFADependent)
    Pinv = inv(model.P)
    C = model.C
    objs = Float64[]
    for (z, w) in zip(zs, ws)
        obj = 0.0
        Lv = model.L(w)
        H = I + Lv'*C'*Pinv*C*Lv
        obj += log(det(H)) + log(det(model.P))
        obj += z'*Pinv*z - z'*Pinv*C*Lv*inv(H)*Lv'*C'*Pinv*z
        push!(objs, real(obj)/length(zs))
    end
    return objs
end


function QGML_objective_comp_parts(zs, ws, model :: MFADependent)
    Pinv = inv(model.P)
    C = model.Ccomp
    objs = Float64[]
    for (z, w) in zip(zs, ws)
        obj = 0.0
        Lv = model.L(w)
        H = I + Lv'*C'*Pinv*C*Lv
        obj += log(det(H)) + log(det(model.P))
        obj += z'*Pinv*z - z'*Pinv*C*Lv*inv(H)*Lv'*C'*Pinv*z
        push!(objs, real(obj)/length(zs))
    end
    return objs
end


function QGML_objective_reduc(zs, ws, model :: MFADependent)
    tot = 0.0
    Pinv = inv(model.P)
    C = model.C
    for (z, w) in zip(zs, ws)
        Lv = model.L(w)
        H = I + Lv'*C'*Pinv*C*Lv
        tot += log(det(H)) - z'*Pinv*C*Lv*inv(H)*Lv'*C'*Pinv*z
    end
    return real(tot)/length(zs)
end


function default_MFA_init(cs, bs)
    model = MFADependent(cs, bs)
    model.A[:] = collect([I(cs.r0); zeros(eltype(model.A), cs.N-cs.r0, cs.r0)])
    for c in 1:cs.C
        model.Bcs[c] = collect([I(cs.rcs[c]); zeros(eltype(model.B), cs.Ncs[c] - cs.rcs[c], cs.rcs[c])])
    end
    blockdiag!(model.B, model.Bcs)
    model.C = [model.A model.B]
    model.P = Diagonal(ones(eltype(model.A), cs.N))
    # initialize to constant specdens function
    init_Ls_seq = [Float64.(collect(I(model.L.totdim)/(2*pi))) for _ in 1:length(bs)]
    init_alpha = packLTsequence(init_Ls_seq, model.L.blockdims)
    model.L = MFACholeskyBSplineFunc(bs, model.L.blockdims, model.L.totdim, Complex.(init_alpha))
    return model
end

function randomloading_MFA_init(rng, cs, bs)
    model = MFADependent(cs, bs)
    model.A = lowertri_invariant(randn(rng, eltype(model.A), cs.N, cs.r0))
    for c in 1:cs.C 
        model.Bcs[c] = lowertri_invariant(randn(rng, eltype(model.B), cs.Ncs[c], cs.rcs[c]))
    end
    blockdiag!(model.B, model.Bcs)
    model.C = [model.A model.B]
    model.P = Diagonal(ones(eltype(model.A), cs.N))
    # initialize to constant identity specdens function
    init_Ls_seq = [Float64.(collect(I(model.L.totdim)/(2*pi))) for _ in 1:length(bs)]
    init_alpha = packLTsequence(init_Ls_seq, model.L.blockdims)
    model.L = MFACholeskyBSplineFunc(bs, model.L.blockdims, model.L.totdim, Complex.(init_alpha))
    return model
end



function randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
    model = randomloading_MFA_init(rng, cs, bs)
    idx=0
    for c in 1:cs.C
        pr = power_ratios_by_channel[c]
        Ac = model.A[(idx+1):(idx+cs.Ncs[c]), :]
        Bc = model.Bcs[c]
        Pc = model.P[(idx+1):(idx+cs.Ncs[c]),(idx+1):(idx+cs.Ncs[c])]
        x = tr(Ac * Ac')
        y = tr(Bc * Bc')
        z = tr(Pc * Pc')
        M = [(1-pr[1])*x -pr[1]*y -pr[1]*z
            -pr[2]*x  (1-pr[2])*y -pr[2]*z
            -pr[3]*x -pr[3]*y  (1-pr[3])*z
        ]
        decomp = svd(M) 
        scales = abs.(decomp.V[:, 3]) # basis for null spaced
        model.A[(idx+1):(idx+cs.Ncs[c]), :] = sqrt(scales[1])*Ac
        model.Bcs[c] = sqrt(scales[2]) * Bc
        model.P[(idx+1):(idx+cs.Ncs[c]),(idx+1):(idx+cs.Ncs[c])] = scales[3] * Pc
        idx += cs.Ncs[c]
    end
    blockdiag!(model.B, model.Bcs)
    model.C = hcat(model.A, model.B)
    return model
end

function fullrandom_MFA_init(rng, cs, bs)
    model = MFADependent(cs, bs)
    model.A = lowertri_invariant(randn(rng, eltype(model.A), cs.N, cs.r0))
    for c in 1:cs.C 
        model.Bcs[c] = lowertri_invariant(randn(rng, eltype(model.B), cs.Ncs[c], cs.rcs[c]))
    end
    blockdiag!(model.B, model.Bcs)
    model.C = [model.A model.B]
    model.P = Diagonal(ones(eltype(model.A), cs.N))
    # initialize to constant identity specdens function
    init_Ls_seq = [unhalfvec(randn(rng,eltype(model.L.coeffs),model.L.totdim^2), model.L.totdim, model.L.totdim) for _ in 1:length(bs)]
    init_alpha = packLTsequence(init_Ls_seq, model.L.blockdims)
    model.L = MFACholeskyBSplineFunc(bs, model.L.blockdims, model.L.totdim, Complex.(init_alpha))
    realdiag_invariant!(model.L)
    identityintegral_invariant!(model.C, model.L, pi*(0.0:1/300:1.0))
  
    extractABcs!(model.A, model.Bcs, model.C)
    blockdiag!(model.B, model.Bcs)
    return model
end

function converged(iter, x, oldx, obj, oldobj ; ftol, xtol, maxiter)
    if iter == 0
        return false
    elseif  (iter >= maxiter)
        println("Iter $iter == Maxiter $maxiter. Terminating.")
        return true
    elseif abs(obj - oldobj) < ftol * oldobj
        println("Objective diff $(abs(obj - oldobj)) < $ftol * $oldobj. Converged.")
        return true
    elseif norm(x - oldx) / length(oldx) < xtol
        println("Coeff diff $(norm(x - oldx)) < $(length(x)) * $xtol. Converged.")
        return true
    end
    return false
end


function time_fit(xs, model; ftol=1e-8, xtol=1e-5, maxiter=10000, Pfloor=1e-3, verbose=false, keeptrace=false)
    T = length(xs)
    X = hcat(xs...)
    ws = 2*pi*rfftfreq(T)
    Z = 1/sqrt(T)*rfft(X, 2)
    zs = [Z[:, t] for t in 1:(size(Z)[2])]
    zs = zs[1:(end-1)] # delete eventually once I switch over to periodic B-splines
    ws = ws[1:(end-1)]
    mod, tr = spectral_fit(zs, ws, model; ftol=ftol, xtol=xtol, maxiter=maxiter, Pfloor=Pfloor, verbose=verbose, keeptrace=keeptrace)
    return (mod, tr)
end

function EM_update_stepwise(zs, ws, model, iter; Pfloor)

end


function EM_update!(zs, ws, model; Pfloor)
    println("Pre-P: $(QGML_objective(zs, ws, model))")
    updateP!(zs, ws, model; Pfloor=Pfloor)
    println("Pre-AB: $(QGML_objective(zs, ws, model))")
    updateAB!(zs, ws, model)
    println("Pre-L: $(QGML_objective(zs, ws, model))")
    updateL!(zs, ws, model)
    println("Pre-Rot: $(QGML_objective(zs, ws, model))")
    # lowertri_invariant!(model.A)
    # for Bc in model.Bcs
    #     lowertri_invariant!(Bc)
    # end
    # blockdiag!(model.B, model.Bcs)
    # model.C = hcat(model.A, model.B)

    #unitloadings_invariant!(model.C, model.L)
    println("Pre-Ident: $(QGML_objective(zs, ws, model))")
    identityintegral_invariant!(model.C, model.L, ws)
    extractABcs!(model.A, model.Bcs, model.C)


    println("Final: $(QGML_objective(zs, ws, model))")
    println("reduc: $(QGML_objective_reduc(zs, ws, model))")
    println("complex: $(QGML_objective_comp(zs, ws, model))")
    return
end

#ZAL (2011, Quasi-newton acceleration for high-dimensional optimization algorithms) 
function QNAccelEM_update!(zs, ws, model, Us, Vs; Pfloor)
    model_up1, model_up2 = deepcopy(model), deepcopy(model)
    EM_update!(zs, ws, model_up1; Pfloor = Pfloor) # F(X_n)
    EM_update!(zs, ws, model_up2; Pfloor = Pfloor) 
    EM_update!(zs, ws, model_up2; Pfloor = Pfloor)# F(F(X_n))
    v0 = packmodel(model)
    v1 = packmodel(model_up1)
    v2 = packmodel(model_up2)

    # shift over past iterates
    Us[2:end] = Us[1:(end-1)]
    Vs[2:end] = Vs[1:(end-1)]
    Us[1] = v1 - v0
    Vs[1] = v2 - v1
    U = hcat(Us...)
    V = hcat(Vs...)
    # update model based on secant info
    if cond(U'*U - U'*V) > 1000
        modupdate = model_up2
        println("initial")
    else
        if length(Us) == 1
            c = -1 * (U'*U/(U'*(V-U)))
            println("c : $(c[1,1])")
            updatev = (1.0-c[1,1])*v1 + c[1,1]*v2
        else
            updatev = v1 - V*inv(U'*U - U'*V)*U'*(v0 - v1)
        end
        println("uv - v1: $(norm(updatev - v1)), uv - v2: $(norm(updatev - v2)), uv - v0: $(norm(updatev - v0))")
        modupdate = unpackmodel(updatev, model.cs, model.L.bs)
        obj2 = QGML_objective(zs, ws, model_up2)
        objupdate = QGML_objective(zs, ws, modupdate)
        if obj2 < objupdate # Correct for descent property.
            modupdate = model_up2
        end
    end    
    model.A = modupdate.A
    model.B = modupdate.B
    model.Bcs = modupdate.Bcs
    model.C = modupdate.C
    model.P = modupdate.P
    model.L = modupdate.L
    return 
end


function spectral_fit(zs, ws, model :: MFADependent; accel_q = 0, ftol=1e-8, xtol=1e-5, maxiter=10000, Pfloor=1e-3, verbose=false, keeptrace=false)
    oldmodel = deepcopy(model)
    T = length(zs)
    if length(ws) != T
        throw(ArgumentError("Number of FFT Obs $T but number of Observation Locations is $(length(ws))"))
    end
    iter = 0
    obj = QGML_objective(zs, ws, model)
    if verbose
        println("Initial objective: $obj")
    end
    oldobj = obj
    trace = [] #(iter num, param values, objective, delta_C norm, delta_P norm, delta_alpha_norm)
    if accel_q > 0
        Us = [zeros(eltype(eltype(zs)), length(packmodel(oldmodel))) for _ in 1:accel_q]
        Vs = [zeros(eltype(eltype(zs)), length(packmodel(oldmodel))) for _ in 1:accel_q]
    end
    while !converged(iter, packmodel(model), packmodel(oldmodel), obj, oldobj; ftol=ftol, xtol=xtol, maxiter=maxiter)
        iter += 1
        # P update
        oldobj = obj
        if keeptrace
            push!(trace, [iter, deepcopy(oldmodel), oldobj, norm(model.C - oldmodel.C), norm(model.P - oldmodel.P), norm(model.L.coeffs - oldmodel.L.coeffs)])
        end
        oldmodel = deepcopy(model)
        if accel_q == 0
            EM_update!(zs, ws, model; Pfloor = Pfloor)
        else
            QNAccelEM_update!(zs, ws, model, Us, Vs; Pfloor = Pfloor)
        end
        obj = QGML_objective(zs, ws, model)
        nm = norm(packmodel(model) - packmodel(oldmodel))
        if verbose
            println("Iter $iter -- Obj: $obj Nm: $nm, Cnm: $(norm(model.C - oldmodel.C)) Pnm: $(norm(model.P - oldmodel.P)) Lnm: $(norm(model.L.coeffs - oldmodel.L.coeffs))")
        end
    end
    if keeptrace
        push!(trace, [iter, deepcopy(model), obj, norm(model.C - oldmodel.C), norm(model.P - oldmodel.P), norm(model.L.coeffs - oldmodel.L.coeffs)])
    end
    return (model, trace)
end


function updateP!(zs, ws, model; Pfloor = 1e-3)
    C, L, P = model.C, model.L, model.P
    Pinv = inv(P)
    CwtCw = Symmetric(C' * Pinv * C)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = Hermitian(Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    ys = [C*Xis[t]*C'*Pinv*zs[t] for t in 1:length(zs)]
    diffs = [z - y for (z, y) in zip(zs, ys)]
    Vs = [Hermitian(diffs[t] * diffs[t]' + C*Xis[t]*C') for t in 1:length(zs)]
    V = mean(Vs)
    Praw = diag(V)
    model.P[:] = Diagonal(max.(real.(Praw), Pfloor))
    return
end


function updateAB_fullprec!(zs, ws, model)
    C, L, P= BigFloat.(model.C), model.L, BigFloat.(model.P)
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zs2 = [Complex{BigFloat}.(z) for z in zs]
    zws2 = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(Complex{BigFloat}, L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = Complex{BigFloat}.(L(w))
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    # Compute least-square mats
    Hs = 1/T*[(Xis[t] + Xis[t]*Cw'*zws2[t]*zws2[t]'*Cw*Xis[t]) for t in 1:length(ws)]
    Us = 1/T*[zs2[t]*zws2[t]'*Cw*Xis[t] for t in 1:length(ws)]
    H = sum(Hs)
    U = sum(Us)
    # Compute new C
    #Cwnew =real(U*inv(H))
    Cnew = real(U*inv(H))#sqrt(P) * Cwnew
    Anew, Bcsnew = extractABcs(Cnew, model.cs.Ncs, model.cs.r0, model.cs.rcs)
    Anew = Float64.(Anew)
    Bcsnew = [Float64.(Bc) for Bc in Bcsnew]
    #lowertri_invariant!(Anew)
    #for Bc in Bcsnew
    #    lowertri_invariant!(Bc)
    #end
    Bnew = blockdiag(Bcsnew)
    model.A = Anew
    model.B = Bnew
    model.Bcs = Bcsnew
    model.C = hcat(Anew, Bnew)
    return
end

function updateAB!(zs, ws, model)
    C, L, P= model.C, model.L, model.P
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    # Compute least-square mats
    Hs = [(Xis[t] + Xis[t]*Cw'*zws[t]*zws[t]'*Cw*Xis[t]) for t in 1:length(ws)]
    Us = [zs[t]*zws[t]'*Cw*Xis[t] for t in 1:length(ws)]
    H  = mean(Hs)
    U = mean(Us)
    # Compute new C
    #Cwnew =real(U*inv(H))
    Ccomp = U  * inv(H)
    Hsq = sqrt(H)
    Psq = sqrt(P)
    Kv = make_K2(Hsq, Psq, model.cs)
    SKv = sparse(Kv)
    K = vcat(real.(SKv), imag(SKv))
    targ = Psq * U * inv(Hsq)
    y = vcat(vec(real.(targ)), vec(imag.(targ)))
    wnew = (K' * K) \ (K' * y)
    Crnew = w_to_C(wnew, model.cs)
    Anew, Bcsnew = extractABcs(Crnew, model.cs.Ncs, model.cs.r0, model.cs.rcs)

    Bnew = blockdiag(Bcsnew)
    model.A = Anew
    model.B = Bnew
    model.Bcs = Bcsnew
    model.C = hcat(Anew, Bnew)
    model.Ccomp = Ccomp
    return
end


function updateAB2!(zs, ws, model)
    C, L, P= model.C, model.L, model.P
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    # Compute least-square mats
    Hs = [(Xis[t] + Xis[t]*Cw'*zws[t]*zws[t]'*Cw*Xis[t]) for t in 1:length(ws)]
    Us = [zs[t]*zws[t]'*Cw*Xis[t] for t in 1:length(ws)]
    H  = mean(Hs)
    U = mean(Us)
    # Compute new C
    #Cwnew =real(U*inv(H))
    # New, Least squares way of computing!!!!!
    
    Ccomp = U  * inv(H)
    Cnew = real(Ccomp)#sqrt(P) * Cwnew
    Anew, Bcsnew = extractABcs(Cnew, model.cs.Ncs, model.cs.r0, model.cs.rcs)

    Bnew = blockdiag(Bcsnew)
    model.A = Anew
    model.B = Bnew
    model.Bcs = Bcsnew
    model.C = hcat(Anew, Bnew)
    model.Ccomp = Ccomp
    return
end



function updateAB_comp!(zs, ws, model)
    C, L, P= model.Ccomp, model.L, model.P
    Pinv = inv(P)
    T = length(ws)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = (Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
    end
    # Compute least-square mats
    Hs = [(Xis[t] + Xis[t]*Cw'*zws[t]*zws[t]'*Cw*Xis[t]) for t in 1:length(ws)]
    Us = [zs[t]*zws[t]'*Cw*Xis[t] for t in 1:length(ws)]
    H  = mean(Hs)
    U = mean(Us)
    # Compute new C
    #Cwnew =real(U*inv(H))
    Ccomp = U  * inv(H)
    Cnew = real(Ccomp)#sqrt(P) * Cwnew
    Anew, Bcsnew = extractABcs(Cnew, model.cs.Ncs, model.cs.r0, model.cs.rcs)

    Bnew = blockdiag(Bcsnew)
    model.A = Anew
    model.B = Bnew
    model.Bcs = Bcsnew
    model.C = hcat(Anew, Bnew)
    model.Ccomp = Ccomp
    return
end

function HQuadratic(L, ws, Hs, CtC)
    tot = 0
    for (t, w) in enumerate(ws)
        tot += tr(Hs[t] * L(w)' * CtC * L(w))
    end
    return tot
end

function UQuadratic(L, ws, Us)
    tot = 0
    for (t, w) in enumerate(ws)
        tot -= 2*real(tr(Us[t]*L(w)'))
    end
    return tot
end

function HUQuadratic(L, ws, Hs, Us, CtC)
    tot = 0
    for (t, w) in enumerate(ws)
        tot += tr(Hs[t] * L(w)' * CtC * L(w))
        tot -= 2*real(tr(Us[t]*L(w)'))
    end
    return tot
end

# Convert trace function to approriate matrix with L.coeffs' * HQ * L.coeffs being the correct val.
function HQuadmat(ws, Hs, CtC, bs, blkdims)
    K = length(bs)
    sz = K * sum([s*(s+1) ÷ 2 for s in blkdims])
    HQ = zeros(eltype(eltype(Hs)), sz, sz)
    idx1 = 0
    # just loop over all valid indices, rows then cols
    for k1 in 1:K
        for c1 in 1:length(blkdims)
            for j1 in 1:blkdims[c1]
                for i1 in j1:blkdims[c1]
                    idx1 += 1
                    idx2 = 0
                    (row1, col1) = getelempos(blkdims, c1-1, j1, i1)
                    for k2 in 1:K
                        for c2 in 1:length(blkdims)
                            for j2 in 1:blkdims[c2]
                                for i2 in j2:blkdims[c2]
                                    idx2 += 1
                                    (row2, col2) = getelempos(blkdims, c2-1, j2, i2) # row2, row1 -- col1 col2
                                    HQ[idx2, idx1] = sum([bs[k1](w)*bs[k2](w)*CtC[row2, row1]*Hs[t][col1, col2] for (t, w) in enumerate(ws)])
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return HQ
end

# construct vector so that trace form is L.coeffs'*Uv
function UVec(ws, Us, bs, blkdims)
    K = length(bs)
    sz = K * sum([s*(s+1) ÷ 2 for s in blkdims])
    Uv = zeros(eltype(eltype(Us)), sz)
    idx = 0
    for k in 1:K
        for c in 1:length(blkdims)
            for j in 1:blkdims[c]
                for i in j:blkdims[c]
                    idx += 1
                    (row, col) = getelempos(blkdims, c-1, j, i)
                    Uv[idx] = sum([bs[k](w)*Us[t][row, col] for (t, w) in enumerate(ws)])
                end
            end
        end
    end
    return Uv
end

function updateL!(zs, ws, model)
    C, L, P= model.C, model.L, model.P
    T = length(ws)
    Pinv = inv(P)
    Psqinv = sqrt(Pinv)
    Cw = Psqinv * C
    zws = [Psqinv * z for z in zs]
    CwtCw = Symmetric(Cw'*Cw)
    Xis = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    Lvs = [zeros(eltype(L.coeffs), L.totdim, L.totdim) for _ in 1:length(ws)]
    for (t, w) in enumerate(ws)
        Lv = L(w)
        Lvs[t] = Lv
        Xis[t] = Hermitian(inv(I + Lv'*CwtCw*Lv)) # Note lack of flanking Lvs
    end
    Hs = 1/T*[Hermitian(Xis[t] + Xis[t]*Lvs[t]'*Cw'*zws[t]*zws[t]'*Cw*Lvs[t]*Xis[t]) for t in 1:length(ws)]
    Us = 1/T*[Cw'*zws[t]*zws[t]'*Cw*Lvs[t]*Xis[t] for t in 1:length(ws)]
    HQ = HQuadmat(ws, Hs, CwtCw, L.bs, L.blockdims)
    Uv = UVec(ws, Us, L.bs, L.blockdims)
    L.coeffs[:] =  (HQ \ Uv)
    realdiag_invariant3!(L) 
    return
end


function get_obs_cov(xs, ws, C, P, fac_specdens_func)
    T = length(xs)
    N = length(xs[1])
    covmats = Matrix{Float64}[]
    for h in 0:(T-1)
        mat = C * real(invert_specdens(h, ws, fac_specdens_func)) * C' 
        if h == 0
            mat+=P
        end
        push!(covmats,mat )
    end
    fullcov = zeros(N*T, N*T)
    idx1 = 0
    for t in 1:T
        idx2 = idx1
        for s in t:T
            fullcov[(idx1+1):(idx1+N), (idx2+1):(idx2+N)] = covmats[s-t+1] 
            idx2+=N
        end
        idx1+=N
    end
    fullcov = Symmetric(fullcov)
    return fullcov
end



function factor_predict(xs, ws, C, P, fac_specdens_func)
    x = vcat(xs...)
    fullcov = get_obs_cov(xs, ws, C, P, fac_specdens_func)
    T = length(xs)
    N = length(xs[1])
    r = size(fac_specdens_func(0.0))[1]
    factor_covmats = Matrix{Float64}[]
    for h in 0:(T-1)
        mat = real(invert_specdens(h, ws, fac_specdens_func))
        push!(factor_covmats,mat )
    end
    fullfaccov = zeros(r*T, N*T)
    idx1 = 0
    for t in 1:(T)
        idx2 = 0
        for s in 1:(T)
            if s >= t
                fullfaccov[(idx1+1):(idx1+r), (idx2+1):(idx2+N)] = factor_covmats[s-t+1] * C'
            else
                fullfaccov[(idx1+1):(idx1+r), (idx2+1):(idx2+N)] = factor_covmats[t-s+1]' * C'
            end
            idx2+=N
        end
        idx1+=r
    end
    p= fullfaccov * inv(fullcov) * x
    preds = [p[(r*j+1):(r*j+r)] for j in 0:(T-1)]
    return preds
end