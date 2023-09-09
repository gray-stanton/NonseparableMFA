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
nobs = 200

model_init = load_object("./badmodel.jld2")
xs = load_object("./badxs.jld2")
zs = load_object("./badzs.jld2")
ws = load_object("./badws.jld2")


A = model_init.A
B = model_init.B
C = model_init.C
P = model_init.P
L = model_init.L
K1 = model_init.Ccomp



obj_old = QGML_objective(zs, ws, model_init)

minobj_old = AB_minorizing_objective(zs, ws, model_init.C, model_init)

model_new = deepcopy(model_init)
updateAB!(zs, ws, model_new)

obj_new = QGML_objective(zs, ws, model_new)

minobj_new = AB_minorizing_objective(zs, ws, model_new.C, model_init)


C, L, P= model_init.C, model_init.L, model_init.P
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

minor = Cn -> norm(Psqinv * U * sqrt(inv(H)) - Psqinv * Cn * sqrt(H))^2
# Prob 1: want to minimize minor(Cn) wrt Cn s.t. Cn real = [An Bn] block structrured.   
# Prob 2: want to minimize minor(Cn) wrt Cn s.t. Cn real, no other structure imposed.


# Fact: U * inv(H) is solution to unconstrained prob, as it's obviously prositive
minor(U * inv(H)) # = 0

# Next, note that previous C is feasible for prob 1(all entries real, block structured)
Cold = model_init.C 
Cold_val = minor(Cold) # 2.3842 > 0

# Also real part of previous Ccomp also feasible for prob 2 (all entries real)
Cold_comp = real(model_init.Ccomp)
Cold_comp_val = minor(BigFloat.(Cold_comp)) # 1.79484


# Now compare to proposed solutions to Prob 1 and Prob 2
# First, for prob 2 propose that taking real part of unconstrained sln is optimal
Cnew_real_part = real(U*inv(H))
Cnew_real_part_val = minor(BigFloat.(Cnew_real_part))
Cnew_real_part_val - Cold_comp_val # small 2.4806e-5, but > 0. 
# problem! this suggests that our proposed solution cannot work, because our old feasible value is strictly less.







# New, using old values
# obj_comp_parts_old = QGML_objective_comp_parts(zs, ws, model_init)

# obj_comp_parts_new = QGML_objective_comp_parts(zs, ws, model_new)


# N1, N2, N3 = AB_minorizing_obj_parts(zs, ws, K1, model_init)

# M1, M2, M3 = AB_minorizing_obj_parts(zs, ws, K2, model_init)

# mindif = N3 - M3
# objdiff = obj_comp_parts_old - obj_comp_parts_new

# logdets1 = [log(det(I + L(w)' * K2' * inv(P) * K2 * L(w))) for w in ws]
# trmins1 = [tr(L(w)*inv(I+L(w)'*C' * inv(P) * ) * K2 * inv(P) * K2) for w in ws]
