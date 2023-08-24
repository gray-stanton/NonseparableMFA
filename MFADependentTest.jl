using LinearAlgebra
using BSplines
using Distributions
using Test


include("./MFADependentUtils.jl")
include("./MFADependentEstim.jl")
include("./MFADependentSample.jl")


rng = MersenneTwister(343)

# Initialization
@testset "initialization" begin
    Ncs = [5, 4]
    rcs = [2, 1]
    r0 = 3
    cs = ChannelSpec(Ncs, rcs, r0)

    @test cs.N == 9
    @test cs.r == 3

    bps = pi*(0.0:0.25:1.0)
    order = 2
    bs = BSplineBasis(order, bps)
    L = MFACholeskyBSplineFunc(bs, vcat(r0, rcs))

    @test L.totdim == 6
    @test length(L.coeffs) == length(bs) * (6 + 3 + 1)

    params = MFADependent(cs, bs)
    @test size(params.A) == (9, 3)
    @test size(params.B) == (9, 3)
    @test length(params.Bcs) == 2
    params.Bcs[1][1,1] = 2.0
    @test params.Bcs[2][1,1] != params.Bcs[1][1,1]

    params2 = default_MFA_init(cs, bs)
    @test params2.A[1, 1] == 1.0
    @test params2.A[2, 2] == 1.0
    @test params2.A[1 ,2] == 0.0
    @test params2.A[4, 1] == 0.0
    @test params2.B[1, 1] == 1.0
    @test params2.B[3, 1] == 0.0
    @test params2.B[3, 3] == 0.0
    @test norm(params2.P - I) <= 1e-10
    @test params2.L.coeffs[1,1] == 1.0/(2*pi)
    @test params2.L.coeffs[2, 1] == 0.0
    @test norm(params2.B - blockdiag(params2.Bcs)) <= 1e-10

    params3 = randomloading_MFA_init(rng, cs, bs)
    @test sign(params3.A[1, 1]) == 1.0
    @test norm(params3.A[1, 2]) <= 1e-14
    @test norm(params3.B[1, 2]) <= 1e-14
    @test norm(params3.B[1, 3]) <= 0.0
    @test norm(params3.P - I) <= 1e-10
    @test params3.L.coeffs[1,1] == 1.0 / (2*pi)
    @test params3.L.coeffs[2, 1] == 0.0
    @test norm(params3.B - blockdiag(params3.Bcs)) <= 1e-10

end

@testset "utility" begin

    b1 = [1.0 2.0; 3.0 4.0]
    b2 = [5.0 6.0 7.0; 8.0 9.0 10.0]
    B = blockdiag([b1, b2])
    @test size(B) == ((2+2), (2+3))
    @test B[1, 2] == 2.0
    @test B[3, 4] == 6.0
    @test unblockdiag(B, [2, 2], [2, 3]) == [b1, b2]

    V = [2.0 0.0; 4.0 6.0]
    @test halfvec(V) == [2.0, 4.0, 6.0]
    @test unhalfvec([2.0, 4.0, 6.0], 2, 2) == V

    A = [1.0 0.0; 2.0 3.0; 4.0 5.0]
    @test halfvec(A) == [1.0, 2.0, 4.0, 3.0, 5.0]
    @test unhalfvec([1.0, 2.0, 4.0, 3.0, 5.0], 3, 2) == A

    bps = pi*(0.0:0.5:1.0)
    order = 2
    bs = BSplineBasis(order, bps)
    K = length(bs)
    blockdims = [2, 2, 1]
    L1 = blockdiag([[1.0 0.0; 2.0 3.0], [4.0 0.0; 5.0 6.0], [7.0][:, :]])
    L2 = 10*L1
    L3 = 10*L2
    alpha = packLTsequence([L1, L2, L3],  blockdims)
    a1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    @test alpha == vcat(a1, 10*a1, 100*a1)
    @test unpackLTsequence(alpha, blockdims, 3) == [L1, L2, L3]

    L = MFACholeskyBSplineFunc(bs, blockdims, sum(blockdims), Complex.(alpha))

    @test getelempos(L.blockdims, 0, 1, 1) == (1, 1)
    @test getelempos(L.blockdims, 0, 1, 2) == (2, 1)
    @test getelempos(L.blockdims, 1, 1, 1) == (3, 3)
    @test getelempos(L.blockdims, 1, 2, 2) == (4, 4)
    @test getelempos(L.blockdims, 2, 1, 1) == (5, 5)
    

    @test getelem(L, 1, 0, 1, 1) == 1.0
    @test getelem(L, 1, 0, 1, 2) == 2.0
    @test getelem(L, 1, 0, 2, 2) == 3.0
    @test getelem(L, 1, 1, 1, 1) == 4.0
    @test getelem(L, 1, 1, 1, 2) == 5.0
    @test getelem(L, 1, 1, 2, 2) == 6.0
    @test getelem(L, 1, 2, 1, 1) == 7.0
    @test getelem(L, 2, 0, 1, 1) == 10.0
    @test getelem(L, 2, 0, 1, 2) == 20.0
    @test getelem(L, 2, 0, 2, 2) == 30.0
    @test getelem(L, 2, 1, 1, 1) == 40.0
    @test getelem(L, 2, 1, 1, 2) == 50.0
    @test getelem(L, 2, 1, 2, 2) == 60.0
    @test getelem(L, 2, 2, 1, 1) == 70.0

    @test norm(L(2.0) - bsplines(bs, 2.0)[2] * L2 - bsplines(bs, 2.0)[3]*L3) < 1e-3

end

@testset "invariants" begin
    A = [-1.0 5.0 3.0; 2.0 3.0 4.0; 3.2 4.3 12; 34.0 23.0 11.0; 343.0 21.0 32.0;]
    A2 = deepcopy(A)
    lowertri_invariant!(A)
    
    @test norm(A*A'- A2 * A2') < 1e-10
    @test abs(A[1, 2]) <= 1e-15
    @test abs(A[1, 3]) <= 1e-15
    @test abs(A[2, 3]) <= 1e-15
    @test sign(A[1, 1]) == 1.0

    bps = pi*(0.0:0.5:1.0)
    order = 2
    bs = BSplineBasis(order, bps)
    L1 = blockdiag([[-1.0+3.0im 0.0; 2.0 3.0+2.5im], [4.0 0.0; 5.0 -6.0], [-7.0+2.5im][:, :]])
    L2 = 1.2*L1
    L3 = 1.4*L2
    K = length(bs)
    blockdims = [2, 2, 1]
    alpha = packLTsequence([L1, L2, L3],  blockdims)
    L = MFACholeskyBSplineFunc(bs, blockdims, sum(blockdims), Complex.(alpha))
    L2 = deepcopy(L)

    realdiag_invariant!(L)
    @test norm(getelem(L, 1, 0, 1, 1) - abs(-1.0+3.0im)) < 1e-14
    @test norm(getelem(L, 1, 2, 1, 1) - abs(-7.0+2.5im))  < 1e-14
    @test norm(L(1.0) * L(1.0)' - L2(1.0) * L2(1.0)') < 1e-10

    C = randn(rng, 10, 5)
    C[:, 1:2] = lowertri_invariant(C[:, 1:2])
    C[1:5, 3:4] = lowertri_invariant(C[1:5, 3:4])
    C[6:10,3:4] .= 0
    C[6:10, 5] = lowertri_invariant(C[6:10, 5:5])
    C[1:5, 5] .= 0
    C2 = deepcopy(C)
    identityintegral_invariant!(C, L, pi*(0.0:(1/200):1.0)) 
    @test norm(C*L(1.0)*L(1.0)'*C'  - C2*L2(1.0)*L2(1.0)'*C2') < 1e-8
    @test sign(C[1,1]) == 1.0
    @test sign(C[2,2]) == 1.0
    @test sign(C[1,3]) == 1.0
    @test sign(C[2,4]) == 1.0
    @test sign(C[6,5]) == 1.0
    @test abs(C[1, 2]) <= 1e-10
    @test abs(C[1, 4]) <= 1e-10
    @test abs(C[6, 4]) <= 1e-10
    ri = zeros(eltype(L.coeffs), L.totdim, L.totdim)
    for w in pi*(0.0:(1/200):1.0)
        ri += 2*pi*real(L(w)*L(w)')/(length(pi*(0.0:(1/200):1.0)))
    end
    @test norm(ri - I(L.totdim)) <= 1e-8

end


@testset "estimation"  begin
    rng = MersenneTwister(3343)
    T = 1200
    n = 5
    zs = [randn(rng, ComplexF64, n) for _ in 1:T]
    ws = [(t*pi/T) for t in 0:(T-1)]

    cs = ChannelSpec([3, 2], [1, 0], 2)
    bps = pi*(0.0:1/2:1.0)
    order=2
    bs = BSplineBasis(order, bps)
    model = randomloading_MFA_init(rng, cs, bs)
    model2 = deepcopy(model)

    @test norm(unpackmodel(packmodel(model), model.cs, model.L.bs).C - model.C) <= 1e-12
    @test norm(unpackmodel(packmodel(model), model.cs, model.L.bs).L(0.5) - model.L(0.5) ) <= 1e-12
    # Objective
    obj1 = QGML_objective_full(zs, ws, model)
    obj2 = QGML_objective(zs, ws, model)
    @test abs(obj1-obj2) <= 1e-10

    # Converged
    @test !converged(0, packmodel(model), packmodel(model), 0.0, 0.0; ftol=0.0, xtol=0.0, maxiter=1000)
    @test converged(1000, packmodel(model), packmodel(model), 1.0, 1.0; ftol=0.0, xtol=0.0, maxiter=1000)
    @test converged(10, packmodel(model), packmodel(model), 1.0, 1.0; ftol=1e-2, xtol=0.0, maxiter=1000)
    @test converged(10, packmodel(model), packmodel(model), 0.0, 1.0; ftol=0.0, xtol=1e-3, maxiter=1000)
    @test !converged(10, [0], [1.0], 0.9, 1.0; ftol=0, xtol=0, maxiter=1000)


    # P Update
    updateP!(zs, ws, model)
    obj3 = QGML_objective(zs, ws, model)
    updateP!(zs, ws, model)
    obj4 = QGML_objective(zs, ws, model)
    @test obj3 <= obj2 
    @test obj4 <= obj3


    # AB Update
    updateAB!(zs, ws, model)
    obj5 = QGML_objective(zs, ws, model)
    updateAB!(zs, ws, model)
    obj6 = QGML_objective(zs, ws, model)
    @test obj5 <= obj4
    @test obj6 <= obj5

    @test minimum(diag(model.P)) > 1e-3
    @test norm(model.A - lowertri_invariant(model.A)) < 1e-10 

    # L Update
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

    coeffnew = (HQ \ Uv)
    @test abs(L.coeffs'*HQ*L.coeffs - HQuadratic(L, ws, Hs, CwtCw)) <= 1e-10
    @test abs(-2*real(L.coeffs'*Uv) - UQuadratic(L, ws, Us)) <= 1e-10
    @test abs(L.coeffs'*HQ *L.coeffs - 2*real(L.coeffs'*Uv) - HUQuadratic(L, ws, Hs, Us, CwtCw)) < 1e-10

    updateL!(zs, ws, model)
    obj7 = QGML_objective(zs, ws, model)
    updateL!(zs, ws, model)
    obj8 = QGML_objective(zs, ws, model)
    @test obj7 <= obj6 
    @test obj8 <= obj7



    # All updates together
    #mod1, trc = spectral_fit(zs, ws, model2; maxiter=500, accel_q=0, verbose=true, ftol=1e-12, keeptrace=true)


    # include("/home/gray/code/NonsepMFAJulia/test_ADA_working.jl")

    # n = 5
    # r = 2
    # T = 500
    
    
    # ws = pi * (0.0:1/T:1.0)
    # bs = BSplineBasis(2, pi*(0.0:1/2:1.0))
    
    # a1 = abs.(randn(length(bs)))
    # a2 = ones(length(bs))
    # a3 = 0.02*randn(ComplexF64, length(bs))
    
    # alpha0 = vcat(a1, a3, a2)
    # L0 = Lspec(2, bs,alpha0)
    
    
    
    # Dsold = get_Ds(L0, ws)
    # rng = MersenneTwister(1232)
    # A = lowertri_rot(randn(n, r))
    # Anew, alphanew = fix_invariant(A, alpha0, length(bs), ws, bs)
    # Lnew = Lspec(2, bs, alphanew)
    # Dsnew = get_Ds(Lnew, ws)
    
    
    # xs = sample_data(Anew, Dsnew, I(n), rng)
    # alpha_init = vcat(ones(ComplexF64, length(bs)), zeros(ComplexF64, length(bs)), ones(ComplexF64, length(bs)))

    # out = fit_L(xs, deepcopy(Anew), I(n), alpha_init, bs, ws; maxiter=500)

    # model3 = deepcopy(model2)
    # model3.A = real(Anew)
    # model3.C = model3.A
    # zs = xs
    # A = Anew
    # W = [1 0 0 0 0 0 0 0 0; 0 0 0 1 0 0 0 0 0; 0 0 0 0 0 0 1 0 0; 0 1 0 0 0 0 0 0 0; 0 0 0 0 1 0 0 0 0; 0 0 0 0 0 0 0 1 0; 0 0 1 0 0 0 0 0 0 ; 0 0 0 0 0 1 0 0 0; 0 0 0 0 0 0 0 0 1]
    # model = model3

    # out2=spectral_fit(xs, ws, model3; maxiter=500, verbose=true, ftol=1e-12, keeptrace=true)

end


@testset "random_estimation" begin
    rng = MersenneTwister(5432)

    T = 2400
    N = 5
    ws = [(t*pi/T) for t in 0:(T-1)]

    cs = ChannelSpec([3, 2], [0, 0], 2)
    bps = pi*(0.0:1/6:1.0)
    order=2
    bs = BSplineBasis(order, bps)

    model0 = randomloading_MFA_init(rng, cs, bs)
    identityintegral_invariant!(model0.C, model0.L, ws)
    extractABcs!(model0.A, model0.Bcs, model0.C)
    blockdiag!(model0.B, model0.Bcs)
    
    model1 = randomloading_MFA_init(rng, cs, bs)
    model2 = deepcopy(model1) 

    zs = sample_zs(rng, ws, model0)

    mod, trc = spectral_fit(zs, ws, model1;accel_q = 1, maxiter=50, verbose=true, ftol=1e-12, keeptrace=false)

    @test norm(mod.A - model0.A)/length(model0.A) <= 0.1
    @test norm(mod.P - model0.P)/N <= 0.1
    @test norm(mod.L.coeffs - model0.L.coeffs)/length(model0.L.coeffs) <= 0.1


    #time domain
    coeffmat = [0.8 0.0; 0.0 0.3]
    innovsd = ident_AR1_innovsd(coeffmat)
    facs = sample_VAR1(rng, T, coeffmat, ident_AR1_innovsd(coeffmat))
    xs = sample_xs(rng, facs, model0.C, model0.P)
    mod2, trc = time_fit(xs, model2; maxiter=250, verbose=true, ftol=1e-12)

    @test norm(mod2.A - model0.A)/length(model0.A) <= 0.1
    @test norm(mod2.P - model0.P)/N <= 0.1
    @test norm(VAR1_specdens(0.4, coeffmat, innovsd) - MFASpecDensBSplineFunc(mod2.L)(0.4)) <= 0.1



end

# S1 = [1.0+3.0im 0.0; 2.0 + 5.0im -3.0+0.5im]
# U =  [exp(-1*im*angle(S1[1,1])) 0.0; 0.0 exp(-1*im*angle(S1[2,2]))]
# S2 = [abs(1.0+3.0im)]

# function unittrans(M)
#     args = angle.(diag(M))
#     U = diagm(exp.(-1*im.*args))
#     return U
# end
