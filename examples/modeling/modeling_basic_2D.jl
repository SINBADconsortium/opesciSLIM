# Example for basic 2D modeling:
# The receiver positions and the source wavelets are the same for each of the four experiments.
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: January 2017
#

using PyCall, PyPlot, opesciSLIM.TimeModeling, SeisIO

## Set up model structure
n = (120,100)	# (x,y,z) or (x,z)
d = (10.,10.)
o = (0.,0.)

# Velocity [km/s]
v = ones(Float32,n) + 0.4f0
v0 = ones(Float32,n) + 0.4f0
v[:,Int(round(end/3)):end] = 2.5f0
v[:,Int(2*round(end/3)):end] = 3.5f0
rho = ones(Float32,n)

# Slowness squared [s^2/km^2]
m = (1./v).^2
m0 = (1./v0).^2
dm = vec(m - m0)

# Setup info and model structure
nsrc = 2	# number of sources
model = Model(n,d,o,m)	# to include density call Model(n,d,o,m,rho)
model0 = Model(n,d,o,m0)

## Set up receiver geometry
nxrec = 120
xrec = linspace(50.,1150.,nxrec)
yrec = 0.
zrec = linspace(50.,50.,nxrec)

# receiver sampling and recording time
timeR = 1000.	# receiver recording time [ms]
dtR = 2.	# receiver sampling interval

# Set up receiver structure
recGeometry = Geometry(xrec,yrec,zrec;dt=dtR,t=timeR,nsrc=nsrc)

## Set up source geometry (cell array with source locations for each shot)
xsrc = convertToCell(linspace(100.,1100.,nsrc))
ysrc = convertToCell(linspace(0.,0.,nsrc))
zsrc = convertToCell(linspace(20.,20.,nsrc))

# source sampling and number of time steps
timeS = 1000.
dtS = 2.	# receiver sampling interval

# Set up source structure
srcGeometry = Geometry(xsrc,ysrc,zsrc;dt=dtS,t=timeS)

# setup wavelet
f0 = 0.01
wavelet = source(timeS,dtS,f0)

# Set up info structure for linear operators
ntComp = get_computational_nt(srcGeometry,recGeometry,model)
info = Info(prod(n),nsrc,ntComp)

######################## WITHOUT DENSITY ############################################

# Options structures
opt = Options(save_data_to_disk=false,
			  file_path=pwd(),	# path to files
			  file_name="shot_record"	# saves files as file_name_xsrc_ysrc.segy
			  )

opt0 = Options(save_data_to_disk=false,
			  file_path=pwd(),	# path to files
			  file_name="linearized_shot_record"	# saves files as file_name_xsrc_ysrc.segy
			  )

# Setup operators
Pr = joProjection(info,recGeometry)
F = joModeling(info,model;options=opt)
F0 = joModeling(info,model0;options=opt0)
Ps = joProjection(info,srcGeometry)
q = joData(srcGeometry,wavelet)

# Combined operator Pr*F*Ps'
Ffull = joModeling(info,model,srcGeometry,recGeometry)

# Nonlinear modeling
d1 = Pr*F*Ps'*q	# equivalent to d = Ffull*q
qad = Ps*F'*Pr'*d1

# Linearized modeling
J = joJacobian(Pr*F0*Ps',q)	# equivalent to J = joSetupJacobian(Ffull,q)
dD = J*dm
rtm = J'*dD

# Plot results
figure(); imshow(d1.data[1], vmin=-1e1, vmax=1e1)
figure(); imshow(reshape(rtm,model.n)', ColorMap("gray"))




