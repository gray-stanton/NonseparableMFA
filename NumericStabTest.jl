cd("/home/gray/code/")
include("/home/gray/code/NonsepMFAJulia/MFADependent.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentUtils.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentEstim.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentSample.jl")


rng = MersenneTwister(1342) #1343
K = 4
bs = BSplineBasis(4, pi*(0.0:1/4:1.0))
Ncs = [20, 15, 10]
rcs = [4, 3, 2]
r0 = 2
cs = ChannelSpec(Ncs, rcs, r0)
power_ratios_by_channel = [[0.3, 0.3, 0.4] for _ in 1:length(Ncs)]


nobs = 100
coeffmat = diagm(repeat([0.8], r0+sum(rcs)))
innovsd = ident_AR1_innovsd(coeffmat)
ws = rfftfreq(nobs)
model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
xs = sample_xs(rng, facs, model0.C, model0.P)
X = hcat(xs...)
ws = 2*pi*rfftfreq(nobs)
Z = 1/sqrt(nobs)*rfft(X, 2)
zs = [Z[:, t] for t in 1:(size(Z)[2])]
println("Run $run.")
model_init = default_MFA_init(cs,bs)
model1, trc = time_fit(xs, model_init; maxiter = 200, verbose=true, keeptrace=true)

model2 = deepcopy(model1)

println("Pre-AB ups $(QGML_objective(zs, ws, model1))")
for q in 1:5
    updateAB!(zs, ws, model)
end

println("Post-AB ups $(QGML_objective(zs, ws, model1))")
