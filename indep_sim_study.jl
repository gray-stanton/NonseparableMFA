include("./MFADependent.jl")
include(".//MFADependentUtils.jl")
include("./MFADependentEstim.jl")
include("./MFADependentSample.jl")
include("./MFAIndependentEstim.jl")


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






function NMSE(model0, model1)
    R0 = model0.C * model0.C' + model0.P
    R1 = model1.C * model1.C' + model1.P
    out = tr((R1-R0)'*(R1-R0))/tr(R0'*R0)
    return out
end

rng = MersenneTwister(1343)
K = 4
bs = BSplineBasis(3, pi*(0.0:1/4:1.0))
Ncs = [6, 8, 10, 12]
rcs = [1, 1, 1, 1]
r0 = 2
cs = ChannelSpec(Ncs, rcs, r0)
power_ratios_by_channel = [[0.1, 0.5, 0.4] for _ in 1:length(Ncs)]


trcs = []
mods = []
itrcs = []
imods = []
model0s = []
refnmse = []
nrun = 4
nobs_vals = [100, 200, 300, 400, 500, 600, 700, 800]
arval = 0.0
coeffmat = diagm(repeat([arval], r0+sum(rcs)))
innovsd = ident_AR1_innovsd(coeffmat)
for nobs in nobs_vals
    for run in 1:nrun
        println("$nobs  Run: $run")
        model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
        facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
        xs = sample_xs(rng, facs, model0.C, model0.P)
        X = hcat(xs...)
        ws = 2*pi*rfftfreq(nobs)
        Z = 1/sqrt(nobs)*rfft(X, 2)
        zs = [Z[:, t] for t in 1:(size(Z)[2])]
        model_init1 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
        model_init2 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
        mod, trc = time_fit(xs, model_init1; maxiter=100, verbose=true, keeptrace=true)
        imod, itrc = fit_indep(xs, model_init2; maxiter=50)
        push!(mods, mod)
        push!(trcs, trc)
        push!(imods, imod)
        push!(itrcs, itrc)
        push!(model0s, model0)
        push!(refnmse, norm(1/nobs * X * X' - model0.C * model0.C' - model0.P)^2/norm(model0.C*model0.C'+model0.P)^2)
    end
end

for m in mods
    m.C = m.C / sqrt(2*pi)
end

nmses = [NMSE(model0s[i], mods[i]) for i in 1:length(mods)]
inmses = [NMSE(model0s[i], imods[i]) for i in 1:length(mods)]

mat = zeros(length(nmses), 6)
idx=0
for nobs in nobs_vals
    for run in 1:nrun
        global idx+=1
        mat[idx, :] = [nmses[idx], inmses[idx], refnmse[idx], nobs, run, arval]
    end
end 
nmsesdf2 = DataFrame(mat, [:depnmse, :indepnmse, :sampcovnmse, :nobs, :run, :arval])
CSV.write("./changing_nobs_NMSE_indep2.csv", nmsesdf2)