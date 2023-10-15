include("./MFADependent.jl")
include(".//MFADependentUtils.jl")
include("./MFADependentEstim.jl")
include("./MFADependentSample.jl")

using CSV
using DataFrames

# B-Spline direct sample


function bspline_sim(bs, cs, nobs, nruns, errsd=1.0, seed = 1232, maxiter=250)
    rng = MersenneTwister(seed)
    model0 = fullrandom_MFA_init(rng, cs, bs)
    ws = pi * (0.0:(1/nobs):1.0)
    model0.P = model0.P * errsd
    zs = sample_zs(rng, ws, model0)
    trcs = []
    mods = []
    for run in 1:nruns
        println("Run $run.")
        model1 = randomloading_MFA_init(rng, cs, bs)
        mod, trc = spectral_fit(zs, ws, model1; maxiter=maxiter, verbose=true)
        push!(trcs, trc)
        push!(mods, mod)
    end
    return (model0, zs, ws, mods, trcs)
end

function var1_sim(bs, cs, nobs, nruns, coeffmat, errsd=1.0, seed=1232, maxiter=250)
    rng = MersenneTwister(1232)
    model0 = randomloading_MFA_init(rng, cs, bs)
    model0.P *= errsd    
    ws = rfftfreq(nobs)
    #identityintegral_invariant!(model0.C, model0.L, ws)
    innovsd = ident_AR1_innovsd(coeffmat)
    facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
    xs = sample_xs(rng, facs, model0.C, model0.P)
    X = hcat(xs...)
    ws = 2*pi*rfftfreq(nobs)
    Z = 1/sqrt(nobs)*rfft(X, 2)
    zs = [Z[:, t] for t in 1:(size(Z)[2])]
    trcs = []
    mods = []
    for run in 1:nruns
        println("Run $run.")
        model1 = randomloading_MFA_init(rng, cs, bs)
        mod, trc = time_fit(xs, model1; maxiter=maxiter, verbose=true)
        push!(trcs, trc)
        push!(mods, mod)
    end
    return (model0, xs, zs, facs, ws, mods, trcs)
end



include("./MFAIndependentEstim.jl")


nobs = 400
arval = 0.95
snrs = [-22, -21, -20, -19, -18, -17, -16, -15]
nrun = 100



# Alternative simulation
rng = MersenneTwister(1343)
K = 4
bs = BSplineBasis(3, pi*(0.0:1/4:1.0))
Ncs = [10, 10]
rcs_a = [1, 1]
r0_a = 1
cs_a = ChannelSpec(Ncs, rcs_a, r0_a)
rcs_b = [1, 2]
r0_b = 0
cs_b = ChannelSpec(Ncs, rcs_b, r0_b)
power_ratios_by_channel = [[0.1, 0.5, 0.4] for _ in 1:length(Ncs)]

#trcs = [] 
#mods = []
itrcs = []
imods = []
like_as = []
like_bs = []
ilike_as = []
ilike_bs = []
singcrits =  []
coeffmat = diagm(repeat([arval], r0_a+sum(rcs_a)))
innovsd = ident_AR1_innovsd(coeffmat)
for snr in snrs
    for run in 1:nrun
        println("$arval  Run: $run")
        phi_01 = diagm(0.01.+rand(rng, Ncs[1]))
        phi_02 = diagm(0.01.+rand(rng, Ncs[2]))
        B_01 = randn(rng, Ncs[1], rcs_a[1])
        B_02 = randn(rng, Ncs[2],rcs_a[2])
        A1 = randn(rng, Ncs[1],r0_a)
        A2 = randn(rng, Ncs[2],r0_a)
        str1 = tr(A1*A1')
        str2 = tr(A2*A2')
        intr1 = tr(B_01*B_01' + phi_01)
        intr2 = tr(B_02*B_02' + phi_02)
        alpha1 = sqrt(10^(snr/10) * intr1 *1/str1)
        alpha2 = sqrt(10^(snr/10) * intr2 *1/str2)
        A_01 = A1 * alpha1
        A_02 = A2 * alpha2
        A0 = vcat(A_01, A_02)
        B0 = blockdiag([B_01, B_02])
        P0 = blockdiag([phi_01, phi_02])
        C0 = hcat(A0, B0)
        facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
        xs = sample_xs(rng, facs, C0, P0)
        X = hcat(xs...)
        T = length(xs)
        S = 1/T * X * X'
        ws = 2*pi*rfftfreq(nobs)
        Z = 1/sqrt(nobs)*rfft(X, 2)
        zs = [Z[:, t] for t in 1:(size(Z)[2])]
        model_init1 = default_MFA_init( cs_a, bs)
        model_init2 = default_MFA_init(cs_b, bs)
        model_init3 = randomloading_MFA_init(rng, cs_a, bs, power_ratios_by_channel)
        model_init4 = default_MFA_init(cs_b, bs)
        mod_a, trc = time_fit(xs, model_init1; maxiter=150, verbose=true, keeptrace=true)
        mod_b, trc = time_fit(xs, model_init2; maxiter=150, verbose=true, keeptrace=true)
        imod_a, itrc = fit_indep(xs, model_init3; maxiter=75)
        imod_b, itrc = fit_indep(xs, model_init4; maxiter=75)

        S11 = S[1:Ncs[1], 1:Ncs[1]]
        S22 = S[(Ncs[1]+1):(Ncs[1]+Ncs[2]), (Ncs[1]+1):(Ncs[1]+Ncs[2])]
        S12 = S[1:Ncs[1], (Ncs[1]+1):(Ncs[1]+Ncs[2])]
        C = sqrt(inv(S11)) * S12 * sqrt(inv(S22))
        sings = svd(C).S
        crit = prod(1/(1-s^2) for s in sings[1:r0_a])

        push!(like_as, -sum(Ncs)/2*log(2*pi)-1/2*QGML_objective(zs, ws, mod_a))
        push!(like_bs,  -sum(Ncs)/2*log(2*pi)-1/2*QGML_objective(zs, ws, mod_b))
        push!(ilike_as, -sum(Ncs)/2 * log(2*pi) -1/2*indep_objective(S, imod_a))
        push!(ilike_bs, -sum(Ncs)/2 * log(2*pi) -1/2*indep_objective(S, imod_b))
        push!(singcrits, crit)
    end
end



mat = zeros(length(like_as), 10)
idx=0
for snr in snrs
    for run in 1:nrun
        global idx+=1
        mat[idx, :] = [like_as[idx], like_bs[idx], ilike_as[idx], ilike_bs[idx], singcrits[idx], snr, run, arval, nobs, 1]
    end
end 
HAtest = DataFrame(mat, [:dep_HA, :dep_H0, :indep_HA, :indep_H0, :singcrit, :snr, :run,:arval,:nobs, :gentype])
CSV.write("./hyptest_alternative.csv", HAtest)




# Null simulation
rng = MersenneTwister(1343)
K = 4
bs = BSplineBasis(3, pi*(0.0:1/4:1.0))
Ncs = [10, 10]
rcs_a = [1, 1]
r0_a = 1
cs_a = ChannelSpec(Ncs, rcs_a, r0_a)
rcs_b = [1, 2]
r0_b = 0
cs_b = ChannelSpec(Ncs, rcs_b, r0_b)
power_ratios_by_channel = [[0.1, 0.5, 0.4] for _ in 1:length(Ncs)]

#trcs = [] 
#mods = []
itrcs = []
imods = []
like_as = []
like_bs = []
ilike_as = []
ilike_bs = []
singcrits =  []

coeffmat = diagm(repeat([arval], sum(rcs_b)))
innovsd = ident_AR1_innovsd(coeffmat)
for snr in snrs
    for run in 1:nrun
        println("$arval  Run: $run")
        phi_01 = diagm(rand(rng, Ncs[1]))
        phi_02 = diagm(rand(rng, Ncs[2]))
        B_01 = randn(rng, Ncs[1], rcs_a[1])
        B_02 = randn(rng, Ncs[2], rcs_a[2])
        A2 = randn(rng, Ncs[2],r0_a)
        str2 = tr(A2*A2')
        intr2 = tr(B_02*B_02' + phi_02)
        alpha2 = sqrt(10^(snr/10) * intr2 *1/str2)
        A_02 = A2 * alpha2
        B0 = blockdiag([B_01, hcat(A_02, B_02)])
        P0 = blockdiag([phi_01, phi_02])
        C0 = B0
        facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
        xs = sample_xs(rng, facs, C0, P0)
        X = hcat(xs...)
        T = length(xs)
        S = 1/T * X * X'
        ws = 2*pi*rfftfreq(nobs)
        Z = 1/sqrt(nobs)*rfft(X, 2)
        zs = [Z[:, t] for t in 1:(size(Z)[2])]
        model_init1 = default_MFA_init( cs_a, bs)
        model_init2 = default_MFA_init(cs_b, bs)
        model_init3 = randomloading_MFA_init(rng, cs_a, bs, power_ratios_by_channel)
        model_init4 = default_MFA_init(cs_b, bs)
        mod_a, trc = time_fit(xs, model_init1; maxiter=150, verbose=true, keeptrace=true)
        mod_b, trc = time_fit(xs, model_init2; maxiter=150, verbose=true, keeptrace=true)
        imod_a, itrc = fit_indep(xs, model_init3; maxiter=75)
        imod_b, itrc = fit_indep(xs, model_init4; maxiter=75)


        S11 = S[1:Ncs[1], 1:Ncs[1]]
        S22 = S[(Ncs[1]+1):(Ncs[1]+Ncs[2]), (Ncs[1]+1):(Ncs[1]+Ncs[2])]
        S12 = S[1:Ncs[1], (Ncs[1]+1):(Ncs[1]+Ncs[2])]
        C = sqrt(inv(S11)) * S12 * sqrt(inv(S22))
        sings = svd(C).S
        crit = prod(1/(1-s^2) for s in sings[1:r0_a])

        push!(like_as, -sum(Ncs)/2*log(2*pi)-1/2*QGML_objective(zs, ws, mod_a))
        push!(like_bs,  -sum(Ncs)/2*log(2*pi)-1/2*QGML_objective(zs, ws, mod_b))
        push!(ilike_as, -sum(Ncs)/2 * log(2*pi) -1/2*indep_objective(S, imod_a))
        push!(ilike_bs, -sum(Ncs)/2 * log(2*pi) -1/2*indep_objective(S, imod_b))
        push!(singcrits, crit)
    end
end



mat = zeros(length(like_as), 10)
idx=0
for snr in snrs
    for run in 1:nrun
        global idx+=1
        mat[idx, :] = [like_as[idx], like_bs[idx], ilike_as[idx], ilike_bs[idx], singcrits[idx], snr, run, arval, nobs, 0]
    end
end 
H0test = DataFrame(mat, [:dep_HA, :dep_H0, :indep_HA, :indep_H0, :singcrit, :snr, :run,:arval,:nobs, :gentype])
CSV.write("./hyptest_null.csv", H0test)
