
export fwi_objective

function fwi_objective(model_full::ModelRho, source::joData, dObs::joData, srcnum::Int64; options=Options())
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito 
 		
	# Load full geometry for out-of-core geometry containers
	typeof(dObs.geometry) == GeometryOOC && (dObs.geometry = Geometry(dObs.geometry))
	typeof(source.geometry) == GeometryOOC && (source.geometry = Geometry(source.geometry))
	
	# for 3D modeling, limit model to area with sources/receivers
	if options.limit_m == true && model_full.n[3] > 1	# only supported for 3D
		model = deepcopy(model_full)
		model = limit_model_to_receiver_area(source.geometry,dObs.geometry,model,options.buffer_size)
	else
		model = model_full
	end

 	# Dimensions for arrays initialization
	if length(model.n) == 2
		nx, nz = model.n
		damp = zeros(Float32,nx, nz)
		dimsflip = (2,1)
		dimsfull = nz+model.nb*2, nx+model.nb*2	# dimensions for devito (z,x)
	else
		nx, ny, nz = model.n
		damp = zeros(Float32,nx, ny, nz)
		dimsflip = (3,2,1)
		dimsfull = nz+model.nb*2, ny+model.nb*2, nx+model.nb*2	# dimensions for devito (z,y,x)
	end

	# Set up model structure
	modelPy = ct.IGrid(model.o, model.d, model.n, sqrt.(1./model.m), rho=model.rho, nbpml=model.nb)
	dtComp = modelPy[:critical_dt]
	h = modelPy[:get_spacing]()
	
	# Initiate source PyObject
	src = ct.IShot()
	tmaxSrc = source.geometry.t[1]
	src[:set_time_axis](dtComp, tmaxSrc)
	numTracesSrc = length(source.geometry.zloc[1])

	# Initiate receiver PyObject
	rec = ct.IShot()
	tmaxRec = dObs.geometry.t[1]
	rec[:set_time_axis](dtComp, tmaxRec)
	numTracesRec = length(dObs.geometry.zloc[1])

	# Extrapolate input data to computational grid
	qIn = (src[:reinterpolateD](reshape(source.data[1], source.geometry.nt[1], numTracesSrc), source.geometry.dt[1], dtComp))'
	if typeof(dObs.data[1]) == SeisIO.SeisCon
		data = convert(Array{Float32,2},dObs.data[1][1].data)
		dObs = joData(dObs.geometry,data)
	end
	dObserved = (rec[:reinterpolateD](reshape(dObs.data[1], dObs.geometry.nt[1], numTracesRec), dObs.geometry.dt[1], dtComp))'

	ntComp = size(dObserved,2)
	ntSrc = Int(trunc(tmaxSrc/dtComp+1))
	ntRec = Int(trunc(tmaxRec/dtComp+1))
	isempty(options.save_rate) ? save_rate=dtComp : save_rate=options.save_rate

	# Set up source coordinates
	src_coords = setup_grid(source.geometry, model.n, model.o)
	src[:set_receiver_pos](src_coords)
	src[:set_shape](ntComp, numTracesSrc)

	# Set up receiver coordinates
	rec_coords = setup_grid(dObs.geometry, model.n, model.o)
	rec[:set_receiver_pos](rec_coords)
	rec[:set_shape](ntComp, numTracesRec)

	# Initiate acoustic modeling object
	Acoustic = ac.JAcoustic_cg(modelPy, rec, src, s_order=options.space_order, tsave=save_rate)
	
	# Initialize dampening array and extend velocity model
	damp = damp_boundary(add_pml(damp, model.nb), model)
	m = add_pml(model.m, model.nb)
	rho = add_pml(model.rho, model.nb)

	# Reshape from Julia to Python array ordering (x,y,z) -> (z,y,x)
	damp = permutedims(damp,dimsflip)
	m = permutedims(m,dimsflip)
	rho = permutedims(rho,dimsflip)

	# Forward modeling to generate synthetic data and background wavefields
	u = zeros(Float32,(prod(dimsfull), 3))
	nsave = trunc(Int,ntComp/(save_rate/dtComp) +1)
	up = zeros(Float32,(prod(dimsfull),nsave))
	dPredicted = zeros(Float32,rec[:shape])'
	op = Acoustic[:Forward](save=true, free_surface=options.free_surface)
	types, args = process_args(op, Dict(:damp=>damp, :m=>m, :rec=>dPredicted, :rec_coords=>rec_coords', :src=>qIn, :src_coords=>src_coords', :u=>u, :rho=>rho, :usave=>up))
	lib = op[:compile]
	status = ccallexpr(lib, Symbol(op[:name]), types, args)

	# zero-padd output if shot record has more samples than source and observed data
	if ntRec > ntComp
		dPredicted = [dPredicted zeros(size(dOut,1),ntRec - ntComp)]
	end
		
	# Data misfit
	argout1 = .5f0*norm(vec(dPredicted) - vec(dObserved),2)^2.f0

	# Backpropagation of data residual
	v = zeros(Float32,(prod(dimsfull), 3))
	residual = dPredicted - dObserved
	gradient = zeros(Float32, size(m))
	opg = Acoustic[:Gradient](free_surface=options.free_surface)
	types, args = process_args(opg, Dict(:damp=>damp, :m=>m, :rec=>residual, :rec_coords=>rec_coords', :u=>up, :grad=>gradient, :v=>v, :rho=>rho))
	lib = opg[:compile]
	status = ccallexpr(lib, Symbol(opg[:name]), types, args)
	argout2 = remove_pml(permutedims(reshape(gradient,dimsfull),dimsflip),model.nb)


	# Extend gradient back to original size
	if options.limit_m==true && model_full.n[3] > 1
		argout2 = extend_gradient(model_full,model,argout2)
	end
	u=[]; v=[]; gc();
	return [argout1; vec(argout2)]
end


