# 2D FWI gradient test with 4 sources
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, opesciSLIM.TimeModeling

## Set up model structure
n = (80,80)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) + 0.4f0
v[:,Int(round(end/2)):end] = 2.5f0
v0 = smooth10(v,n)

# Slowness squared [s^2/km^2]
m = (1f0./v).^2
m0 = (1f0./v0).^2
dm = m0 - m

# Setup info and model structure
nsrc = 1	# number of sources
ntComp = 250
info = Info(prod(n), nsrc, ntComp)	# number of gridpoints, number of experiments, number of computational time steps
model = Model(n,d,o,m)
model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 101
xrec = linspace(50f0,750f0,nxrec)
yrec = 0.
zrec = linspace(40f0,40f0,nxrec)

# receiver sampling and recording time
timeR = 1200f0	# receiver recording time [ms]
dtR = calculate_dt(n,d,o,v)	# receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell([400f0])
ysrc = convertToCell([0f0])
zsrc = convertToCell([20f0])

# source sampling and number of time steps
timeS = 1200f0
dtS = calculate_dt(n,d,o,v)	# receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.01f0
wavelet = source(timeS,dtS,f0)

###################################################################################################

# Gradient test
h = 1.f-1
iter = 10
error1 = zeros(iter)
error2 = zeros(iter)
h_all = zeros(iter)
srcnum = 1:nsrc
modelH = deepcopy(model0)

# Observed data
F = joModeling(info,model,srcGeometry,recGeometry)
q = joData(srcGeometry,wavelet)
d = F*q

# FWI gradient and function value for m0
Jm0, grad = fwi_objective(model0,q,d)

for j=1:iter
	# FWI gradient and function falue for m0 + h*dm
	modelH.m = model0.m + h*dm
	Jm, gradm = fwi_objective(modelH,q,d)

	dJ = dot(grad,vec(dm))

	# Check convergence
	error1[j] = abs(Jm - Jm0)
	error2[j] = abs(Jm - (Jm0 + h*dJ))

	println(h, " ", error1[j], " ", error2[j])
	h_all[j] = h
	h = h/2f0
end

# Plot errors
loglog(h_all, error1); loglog(h_all, 1e2*h_all)
loglog(h_all, error2); loglog(h_all, 1e2*h_all.^2)
legend([L"$\Phi(m) - \Phi(m0)$", "1st order", L"$\Phi(m) - \Phi(m0) - \nabla \Phi \delta m$", "2nd order"], loc="lower right")
xlabel("h")
ylabel(L"Error $||\cdot||^\infty$")
title("FWI gradient test")
#axis((h_all[end], h_all[1], 1.0e-8,500))


