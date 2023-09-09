cd("/home/gray/code/")
include("/home/gray/code/NonsepMFAJulia/MFADependent.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentUtils.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentEstim.jl")
include("/home/gray/code/NonsepMFAJulia/MFADependentSample.jl")

using Quadmath
using JLD2

rng = MersenneTwister(1343) #1343
K = 4
bs = BSplineBasis(4, (pi+0.03)*(0.0:1/4:1.0))
Ncs = [20, 15, 10]
rcs = [4, 2, 1]
r0 = 2
cs = ChannelSpec(Ncs, rcs, r0)
power_ratios_by_channel = [[0.3, 0.3, 0.2] for _ in 1:length(Ncs)]

trcs = []
mods = []
nrun = 2
nobs = 100

coeffmat = diagm(repeat([0.5], r0+sum(rcs)))
innovsd = ident_AR1_innovsd(coeffmat)
ws = 2 * pi*rfftfreq(nobs)
model0 = randomloading_MFA_init(rng, cs, bs, power_ratios_by_channel)
facs = sample_VAR1(rng, nobs, coeffmat, innovsd)
xs = sample_xs(rng, facs, model0.C, model0.P)
X = hcat(xs...)
Z = 1/sqrt(nobs)*rfft(X, 2)
zs = [Z[:, t] for t in 1:(size(Z)[2])]

zs = zs[1:(end-1)]
ws = ws[1:(end-1)]

save_object("./badxs.jld2", xs)
save_object("./badzs.jld2", zs)
save_object("./badws.jld2", ws)


println("Run $run.")


model_init = fullrandom_MFA_init(rng, cs,bs)
#model_init = load_object("./badmodel.jld2")

# If L(w) is constant (even complex), definitely minorizes.
# al = packLTsequence(repeat(
#    [unpackLTsequence(model_init.L.coeffs, model_init.L.blockdims, length(bs))[end]], 
#     length(bs)), model_init.L.blockdims)
# model_init.L = MFACholeskyBSplineFunc(bs, model_init.L.blockdims, model.L.totdim, al)

#model_init = load_object("./badmodel.jld2")
#model_init.L = load_object("./Lbad.jld2")
#model_init.C = load_object("./Cbad.jld2")
#model_init.P = load_object("./Pbad.jld2")
model_init2 = deepcopy(model_init)
model1, trc = time_fit(xs, model_init; maxiter = 400, verbose=true, keeptrace=true)
#model3, trc = spectral_fit(zs, ws, model_init2; maxiter=100)

save_object("./badmodel.jld2", model1)


Lbad = model1.L
model = model2
model2 = deepcopy(model1)
model3 = deepcopy(model1)
updateAB!(zs, ws, model3)
updateAB_fullprec!(zs, ws, model2)
oldobj = QGML_objective(zs, ws, model1)
testobj = QGML_objective(zs, ws, model3)
testobj2 = QGML_objective(zs, ws, model2)

println("oldobj = $oldobj,  testobj = $testobj, testobj2 = $testobj2")

# #test quadmath.
# C, L, P= BigFloat.(model2.C), model2.L, BigFloat.(model2.P)
# Pinv = inv(P)
# T = length(ws)
# Psqinv = sqrt(Pinv)
# Cw = Psqinv * C
# zs2 = [Complex{BigFloat}.(z) for z in zs]
# zws2 = [Psqinv * z for z in zs2]
# CwtCw = (Cw'*Cw)
# Xis = [zeros(Complex{BigFloat}, L.totdim, L.totdim) for _ in 1:length(ws)]
# for (t, w) in enumerate(ws)
#     Lv = Complex{BigFloat}.(L(w))
#     Xis[t] = (Lv * inv(I + Lv'*CwtCw*Lv) * Lv')
# end
# # Compute least-square mats
# Hs = 1/T*[(Xis[t] + Xis[t]*Cw'*zws[t]*zws[t]'*Cw*Xis[t]) for t in 1:length(ws)]
# Us = 1/T*[zs2[t]*zws2[t]'*Cw*Xis[t] for t in 1:length(ws)]
# H = sum(Hs)
# U = sum(Us)
# # Compute new C
# #Cwnew =real(U*inv(H))
# Cnew = real(U*inv(H))#sqrt(P) * Cwnew
# Anew, Bcsnew = extractABcs(Cnew, model.cs.Ncs, model.cs.r0, model.cs.rcs)
# #lowertri_invariant!(Anew)
# #for Bc in Bcsnew
# #    lowertri_invariant!(Bc)
# #end
# Bnew = blockdiag(Bcsnew)
# model2.A = Anew
# model2.B = Bnew
# model2.Bcs = Bcsnew
# model2.C = hcat(Anew, Bnew)
# newobj = QGML_objective(zs, ws, model2)
# tot = 0.0
# C = hcat(Anew, Bnew)
# for (z, w) in zip(zs, ws)
#     Lv = model.L(w)
#     H = I + Lv'*C'*Pinv*C*Lv
#     tot += log(det(H)) + log(det(model.P))
#     tot += z'*Pinv*z - z'*Pinv*C*Lv*inv(H)*Lv'*C'*Pinv*z
# end
# tot = real(tot)/length(zs)

# test in quad math

# for (i, w) in enumerate(ws[1:(end)])
#     testmodel = deepcopy(model1)
#     onew = [w]
#     onez = [zs[i]]
#     oldobj = QGML_objective(onez, onew, model1)
#     updateAB!(onez, onew, testmodel)
#     newobj = QGML_objective(onez, onew, testmodel)
#     println("OldObj: $oldobj   NewObj: $newobj diff: $(newobj - oldobj)")
# end


println("Pre-AB ups $(QGML_objective(zs, ws, model1))")
for q in 1:5
    updateAB!(zs, ws, model2)
end

println("Post-AB ups $(QGML_objective(zs, ws, model1))")


# Test using old code?
