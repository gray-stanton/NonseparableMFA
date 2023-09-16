include("./MFADependent.jl")
include(".//MFADependentUtils.jl")
include("./MFADependentEstim.jl")
include("./MFADependentSample.jl")

using CSV
using DataFrames



rng = MersenneTwister(1343)
K = 4
bs = BSplineBasis(3, pi*(0.0:1/4:1.0))
Ncs = [15, 15]
rcs = [1, 1]
r0 = 2

trcs = []
mods = []
itrcs = []
imods = []
model0s = []
prats = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
arval = 0.95
coeffmat = diagm(repeat([arval], r0+sum(rcs)))
innovsd = ident_AR1_innovsd(coeffmat)


nrun = 1
nobs=400

nms_s = Float64[]
inms_s = Float64[]
prat_vals = Float64[]
for prat in prats
    for run in 1:nrun
        println("prat $prat run $run")
        power_ratios_by_channel = [[prat, 0.5*(1-prat), 0.5*(1-prat)] for _ in 1:length(Ncs)]
        push!(prat_vals, prat)
        cs = ChannelSpec(Ncs, rcs, r0)
        model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
        coeffmat = diagm(repeat([arval], r0+sum(rcs)))
        innovsd = ident_AR1_innovsd(coeffmat)
        facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
        xs = sample_xs(rng, facs, model0.C, model0.P)
        X = hcat(xs...)
        ws = 2*pi*rfftfreq(nobs)
        Z = 1/sqrt(nobs)*rfft(X, 2)
        zs = [Z[:, t] for t in 1:(size(Z)[2])]
        model_init1 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
        model_init2 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
        depmod, trc = time_fit(xs, model_init1; maxiter=100, verbose=true, keeptrace=true)
        imod, itrc = fit_indep(xs, model_init2; maxiter=50)

        depmod.C = depmod.C / sqrt(2*pi)
        depmod.A = depmod.A / sqrt(2*pi)

        facpreds = factor_predict(xs, pi*(0.0:1/400:1.0), depmod.C, depmod.P, MFASpecDensBSplineFunc(depmod.L))
        ifacpreds = [imod.C'*inv(imod.C*imod.C'+imod.P) * x for x in xs]
        
        spreds =  [depmod.A* fp[1:r0] for fp in facpreds]  
        ispreds = [imod.A* fp[1:r0] for fp in ifacpreds]
        true_s = [model0.A * f[1:r0] for f in facs] 

        nms = mean([norm(sp - ts)^2/norm(ts)^2 for (sp, ts) in zip(spreds, true_s)])
        inms = mean([norm(isp - ts)^2/norm(ts)^2 for (isp, ts) in zip(ispreds, true_s)])
        push!(nms_s, nms)
        push!(inms_s , inms)
    end
end



mat = zeros(length(nms_s), 6)
idx=0
for prat in prats
    for run in 1:nrun
        global idx+=1
        mat[idx, :] = [nms_s[idx], inms_s[idx], nobs, run, arval, prat]
    end
end 
nmsesdf2 = DataFrame(mat, [:depnmse, :indepnmse, :nobs, :run, :arval, :prat])
CSV.write("./signal_recon_powerratio.csv", nmsesdf2)