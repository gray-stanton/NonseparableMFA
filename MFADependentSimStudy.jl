cd("/home/gray/code/")
include("/home/gray/code/NonsepMFAJulia/MFADependentUtils.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentEstim.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentSample.jl")



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



# Graph 1, convergence
# C=3, Ncs = [20, 15, 10], Rcs = [4, 3, 2], r0= 2
# Power ratios are 0.3, 0.3, 0.4 resp.
# T = 100
using Plots
using CSV
using DataFrames

include("/home/gray/code/NonsepMFAJulia/MFAIndependentEstim.jl")

# rng = MersenneTwister(1343)
# K = 4
# bs = BSplineBasis(4, pi*(0.0:1/4:1.0))
# Ncs = [20, 15, 10]
# rcs = [4, 3, 2]
# r0 = 2
# cs = ChannelSpec(Ncs, rcs, r0)
# power_ratios_by_channel = [[0.3, 0.3, 0.4] for _ in 1:length(Ncs)]

# trcs = []
# mods = []
# nrun = 15
# nobs = 100
# coeffmat = diagm(repeat([0.8], r0+sum(rcs)))
# innovsd = ident_AR1_innovsd(coeffmat)
# ws = rfftfreq(nobs)
# for run in 1:nrun
#     model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#     facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
#     xs = sample_xs(rng, facs, model0.C, model0.P)
#     X = hcat(xs...)
#     ws = 2*pi*rfftfreq(nobs)
#     Z = 1/sqrt(nobs)*rfft(X, 2)
#     zs = [Z[:, t] for t in 1:(size(Z)[2])]
#     println("Run $run.")
#     model_init = default_MFA_init(cs, bs)
#     mod, trc = time_fit(xs, model_init; maxiter = 30, verbose=true, keeptrace=true)
#     push!(mods, mod)
#     push!(trcs, trc)
# end
# objs = zeros(nrun, length(trcs[1]))
# for r in 1:nrun
#     for i in 1:length(trcs[1])
#         objs[r, i] = trcs[r][i][3]
#     end
# end

# objsdf = DataFrame( collect(objs'), :auto)
# CSV.write("./obj_by_iteration.csv", objsdf)


function NMSE(model0, model1)
    R0 = model0.C * model0.C' + model0.P
    R1 = model1.C * model1.C' + model1.P
    out = tr((R1-R0)'*(R1-R0))/tr(R0'*R0)
    return out
end

# Fig 2 - NMSE for varying T
# Ncs = [6, 8, 10, 12], r0 = 2, rcs = [1, 1, 1, 1], prop = [0.1, 0.5, 0.4]

rng = MersenneTwister(1343)
K = 4
bs = BSplineBasis(4, pi*(0.0:1/4:1.0))
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
nrun = 250
nobs_vals = [100, 200, 300, 400, 500, 600]
arval = 0.9
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
nmsesdf = DataFrame(mat, [:depnmse, :indepnmse, :sampcovnmse, :nobs, :run, :arval])
CSV.write("./changing_nobs_NMSE2.csv", nmsesdf)


 ## independence.
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
nrun = 250
nobs_vals = [100, 200, 300, 400, 500, 600]
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



# # Fig 3 - Varying AR level

# trcs = []
# mods = []
# itrcs = []
# imods = []
# model0s = []
# refnmse = []
# nrun = 250
# nobs = 400
# ar_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 0.95]
# for arval = ar_vals
#     coeffmat = diagm(repeat([arval], r0+sum(rcs)))
#     innovsd = ident_AR1_innovsd(coeffmat)
#     for run in 1:nrun
#         println("$arval  Run: $run")
#         model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
#         xs = sample_xs(rng, facs, model0.C, model0.P)
#         X = hcat(xs...)
#         ws = 2*pi*rfftfreq(nobs)
#         Z = 1/sqrt(nobs)*rfft(X, 2)
#         zs = [Z[:, t] for t in 1:(size(Z)[2])]
#         model_init1 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         model_init2 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         mod, trc = time_fit(xs, model_init1; maxiter=100, verbose=true, keeptrace=true)
#         imod, itrc = fit_indep(xs, model_init2; maxiter=50)
#         push!(mods, mod)
#         push!(trcs, trc)
#         push!(imods, imod)
#         push!(itrcs, itrc)
#         push!(model0s, model0)
#         push!(refnmse, norm(1/nobs * X * X' - model0.C * model0.C' - model0.P)^2/norm(model0.C*model0.C'+model0.P)^2)
#     end
# end
    


# for m in mods
#     m.C = m.C / sqrt(2*pi)
# end

# nmses = [NMSE(model0s[i], mods[i]) for i in 1:length(mods)]
# inmses = [NMSE(model0s[i], imods[i]) for i in 1:length(mods)]

# mat = zeros(length(nmses), 6)
# idx=0
# for arval in ar_vals
#     for run in 1:nrun
#         idx+=1
#         mat[idx, :] = [nmses[idx], inmses[idx], refnmse[idx], arval, run, nobs]
#     end
# end 
# nmsesdf3 = DataFrame(mat, [:depnmse, :indepnmse, :sampcovnmse, :arval, :run, :nobs])
# CSV.write("./changing_AR_nmsesdf.csv", nmsesdf3)


# # Factor Prediction plot

# rng = MersenneTwister(1343)
# K = 4
# bs = BSplineBasis(3, pi*(0.0:1/4:1.0))
# Ncs = [5, 5]
# rcs = [1, 1]
# r0 = 1
# power_ratios_by_channel = [[0.3, 0.3, 0.4] for _ in 1:length(Ncs)]

# cs = ChannelSpec(Ncs, rcs, r0)
# model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
# nobs = 600
# arval = 0.95
# coeffmat = diagm(repeat([arval], r0+sum(rcs)))
# innovsd = ident_AR1_innovsd(coeffmat)
# facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
# xs = sample_xs(rng, facs, model0.C, model0.P)
# X = hcat(xs...)
# ws = 2*pi*rfftfreq(nobs)
# Z = 1/sqrt(nobs)*rfft(X, 2)
# zs = [Z[:, t] for t in 1:(size(Z)[2])]
# model_init1 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
# model_init2 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
# depmod, trc = time_fit(xs, model_init1; maxiter=100, verbose=true, keeptrace=true)
# imod, itrc = fit_indep(xs, model_init2; maxiter=50)

# depmod.C = depmod.C / sqrt(2*pi)
# facpreds = factor_predict(xs, pi*(0.0:1/1000.0:1.0), depmod.C, depmod.P, MFASpecDensBSplineFunc(depmod.L))
# ifacpreds = [imod.C'*inv(imod.C*imod.C'+imod.P) * x for x in xs]
# true_fac_spec_dens = w -> diagm(repeat([AR1_specdens(w, arval, sqrt(1-arval^2))], r0+sum(rcs)))
# bfacpreds = factor_predict(xs, pi*(0.0:1/1000.0:1.0), model0.C, model0.P, true_fac_spec_dens)
# F = hcat(facs...)
# FP = hcat(facpreds...)
# IFP = hcat(ifacpreds...)
# BFP = hcat(bfacpreds...) 

# R1 = factor_closest_rot(F, FP)
# R2 = factor_closest_rot(F, IFP)
# R3 = factor_closest_rot(F, BFP)

# norm(F - R1 * FP)^2 / norm(F)^2
# norm(F - R2 * IFP)^2 / norm(F)^2
# norm(F - R3 * BFP)^2 / norm(F)^2


# df = DataFrame(collect(hcat(F', (R1*FP)', (R2*IFP)')), [:rf1, :rf2, :rf3, :fp1, :fp2, :fp3, :ifp1, :ifp2, :ifp3])
# CSV.write("./factor_predictions.csv", df)


# # Combined factor predictive error

# rng = MersenneTwister(1343)
# K = 4
# bs = BSplineBasis(3, pi*(0.0:1/4:1.0))
# Ncs = [5, 5]
# rcs = [1, 1]
# r0 = 1
# power_ratios_by_channel = [[0.3, 0.3, 0.4] for _ in 1:length(Ncs)]

# cs = ChannelSpec(Ncs, rcs, r0)

# nobs_vals = [100, 200, 300, 400, 500, 600]

# nrun = 5

# nmse = []
# inmse = []
# arval = 0.8
# coeffmat = diagm(repeat([arval], r0+sum(rcs)))
# innovsd = ident_AR1_innovsd(coeffmat)
# for nobs in nobs_vals
#     for run in 1:nrun
#         model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
#         xs = sample_xs(rng, facs, model0.C, model0.P)
#         X = hcat(xs...)
#         ws = 2*pi*rfftfreq(nobs)
#         Z = 1/sqrt(nobs)*rfft(X, 2)
#         zs = [Z[:, t] for t in 1:(size(Z)[2])]
#         model_init1 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         model_init2 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         modf, trc = time_fit(xs, model_init1; maxiter=140, verbose=true, keeptrace=true)
#         imod, itrc = fit_indep(xs, model_init2; maxiter=70)

#         modf.C = modf.C / sqrt(2*pi)
#         facpreds = factor_predict(xs, pi*(0.0:1/1000.0:1.0), modf)
#         ifacpreds = [imod.C'*inv(imod.C*imod.C'+imod.P) * x for x in xs]
#         F = hcat(facs...)
#         FP = hcat(facpreds...) * (2*pi)
#         IFP = hcat(ifacpreds...)

#         R1 = factor_closest_rot(F, FP)
#         R2 = factor_closest_rot(F, IFP)

#         f1=norm(F - R1 * FP)^2 / norm(F)^2
#         f2=norm(F - R2 * IFP)^2 / norm(F)^2
#         push!(nmse, f1)
#         push!(inmse, f2)
#     end
# end

# mat = zeros(length(nmse), 5)
# idx = 0
# for run in 1:nrun
#     for nobs in nobs_vals
#         idx += 1
#         mat[idx, :] = [nmse[idx], inmse[idx], nobs, nrun, arval]
#     end
# end

# ferrordf = DataFrame(mat, [:fnmse, :ifnmse, :nobs, :nrun, :arval])
# CSV.write("./factor_errors.csv", ferrordf)
# #model0.L = modf.L
# #tfacpreds = factor_predict(xs, pi*(0.0:1/1000.0:1.0), model0)
# #TFP = hcat(tfacpreds...)



# # Fig 7 - Temporal Covariance Estimation


# rng = MersenneTwister(1343)
# K = 4
# bs = BSplineBasis(4, pi*(0.0:1/4:1.0))
# Ncs = [10, 10]
# rcs = [1, 1]
# r0 = 2
# cs = ChannelSpec(Ncs, rcs, r0)
# power_ratios_by_channel = [[0.3, 0.3, 0.4] for _ in 1:length(Ncs)]



# function l2_err(modspecdens, truespecdens, ws) 
#     tot = 0.0
#     for w in ws
#         tot += norm(true_spec_dens(w) - modspecdens(w))^2
#     end
#     return sqrt(tot/length(ws))
# end

# function max_error(modspecdens, truespecdens, ws)
#     out = 0.0
#     for w in ws
#         out = max(out, norm(true_spec_dens(w) - modspecdens(w)))
#     end
#     return out
# end


# trcs = []
# mods = []
# model0s = []
# nrun = 40
# nobs_vals = [100, 300, 500, 700]
# arvals = [0.8, 0.0, 0.5, 0.5]
# coeffmat = diagm(arvals)
# innovsd = ident_AR1_innovsd(coeffmat)
# spfunc = w -> diagm([AR1_specdens(w, coeffmat[i,i], innovsd[i,i]) for i in 1:length(arvals)])
# Lbest = best_fit_chol(spfunc, bs, vcat([r0], rcs...), pi*(0.0:(1/500):1.0))
# spbest = MFASpecDensBSplineFunc(Lbest)

# for nobs in nobs_vals
#     for run in 1:nrun
#         println("$nobs  Run: $run")
#         model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
#         xs = sample_xs(rng, facs, model0.C, model0.P)
#         X = hcat(xs...)
#         ws = 2*pi*rfftfreq(nobs)
#         Z = 1/sqrt(nobs)*rfft(X, 2)
#         zs = [Z[:, t] for t in 1:(size(Z)[2])]
#         model_init1 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         #model_init2 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
#         mod, trc = time_fit(xs, model_init1; maxiter=100, verbose=true, keeptrace=true)

#         #imod, itrc = fit_indep(xs, model_init2; maxiter=50)
#         push!(mods, mod)
#         push!(trcs, trc)
#         push!(model0s, model0)
#     end
# end

# for m in mods
#     m.C = m.C / sqrt(2*pi)
# end

# modspecdens = [MFASpecDensBSplineFunc(mods[i].L) for i in 1:length(mods)]



# l2errs = [sqrt(2)*l2_err(m, spfunc, pi*(0.0:1/1000.0:1.0)) for m in modspecdens]
# linferrs = [max_error(m, spfunc, pi*(0.0:1/1000.0:1.0)) for m in modspecdens]
# l2best = sqrt(2)*l2_err(spbest, spfunc, pi*(0.0:1/1000.0:1.0))
# linfbest = max_error(spbest, spfunc, pi*(0.0:1/1000.0:1.0)) 



# mat = zeros(length(l2errs), 6)
# idx=0
# for nobs in nobs_vals
#     for run in 1:nrun
#         idx+=1
#         mat[idx, :] = [l2errs[idx], linferrs[idx], l2best, linfbest, nobs, run]
#     end
# end 
# temperrdf = DataFrame(mat, [:l2err, :linferr, :l2best, :linfbest, :nobs, :run])
# CSV.write("./temporal_nmse.csv", temperrdf)

# # Fig 8, covariances


# fit_spd =  MFASpecDensBSplineFunc(mods[90].L)
# covs = [2*pi*invert_specdens(h, pi*(0.0:1/1000:1.0), fit_spd) for h in 0:20]
# truecovs = [VAR1_covariance(h, coeffmat, innovsd) for h in 0:20]
# covoutdf = DataFrame(truecov1 = [c[1,1] for c in truecovs], truecov2 = [c[2,2] for c in truecovs], 
#     truecov3 = [c[3, 3] for c in truecovs], ftcov1 = [c[1,1] for c in covs], ftcov2 = [c[2,2] for c in covs],
#      ftcov3 = [c[3,3] for c in covs])
# CSV.write("./covfits.csv", covoutdf)