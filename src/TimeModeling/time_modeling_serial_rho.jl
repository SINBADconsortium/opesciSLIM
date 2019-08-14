
time_modeling(model::ModelRho, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, dm, srcnum::Int64, op::Char, mode::Int64) = 
	time_modeling(model, srcGeometry, srcData, recGeometry, recData, dm, srcnum, op, mode, Options())

function time_modeling(model_full::ModelRho, srcGeometry::Geometry, srcData, recGeometry::Geometry, recData, dm, srcnum::Int64, op::Char, mode::Int64, options)
# Setup time-domain linear or nonlinear foward and adjoint modeling and interface to OPESCI/devito 

	# Load full geometry for out-of-core geometry containers
	typeof(recGeometry) == GeometryOOC && (recGeometry = Geometry(recGeometry))
	typeof(srcGeometry) == GeometryOOC && (srcGeometry = Geometry(srcGeometry))

	# for 3D modeling, limit model to area with sources/receivers
	if options.limit_m == true && model_full.n[3] > 1	# only supported for 3D
		model = deepcopy(model_full)
		if op=='J' && mode==1
			model,dm = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size;pert=dm)
		else
			model = limit_model_to_receiver_area(srcGeometry,recGeometry,model,options.buffer_size)
		end
	else
		model = model_full
	end

	# Dimensions for arrays initialization
	if length(model.n) == 2
		nx, nz = model.n
		damp = zeros(Float32,nx,nz)
		dimsflip = (2,1)
		dimsfull = nz+model.nb*2, nx+model.nb*2	# dimensions for devito (z,x)
	else
		nx, ny, nz = model.n
		damp = zeros(Float32,nx,ny,nz)
		dimsflip = (3,2,1)
		dimsfull = nz+model.nb*2, ny+model.nb*2, nx+model.nb*2	# dimensions for devito (z,y,x)
	end

	# Set up model structure
	modelPy = ct.IGrid(model.o, model.d, model.n, sqrt.(1./model.m), rho=model.rho, nbpml=model.nb)
	dtComp = modelPy[:critical_dt]
	h = modelPy[:get_spacing]()
	
	# Initiate source PyObject
	src = ct.IShot()
	tmaxSrc = srcGeometry.t[1]
	src[:set_time_axis](dtComp, tmaxSrc)
	numTracesSrc = length(srcGeometry.zloc[1])

	# Initiate receiver PyObject
	rec = ct.IShot()
	tmaxRec = recGeometry.t[1]
	rec[:set_time_axis](dtComp, tmaxRec)
	numTracesRec = length(recGeometry.zloc[1])

	# Extrapolate input data to computational grid
	if mode==1
		qIn = (src[:reinterpolateD](reshape(srcData[1], srcGeometry.nt[1], numTracesSrc), srcGeometry.dt[1], dtComp))'
		ntComp = size(qIn,2)
	elseif op=='F' &&  mode==-1
		if typeof(recData[1]) == SeisIO.SeisCon
			recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
		elseif typeof(recData[1]) == String
			recData = load(recData[1])["d"].data
		end
		dIn = (rec[:reinterpolateD](reshape(recData[1], recGeometry.nt[1], numTracesRec), recGeometry.dt[1], dtComp))'
		ntComp = size(dIn,2)
	elseif op=='J' && mode==-1
		if typeof(recData[1]) == SeisIO.SeisCon
			recDataCell = Array{Any}(1); recDataCell[1] = convert(Array{Float32,2},recData[1][1].data); recData = recDataCell
		elseif typeof(recData[1]) == String
			recData = load(recData[1])["d"].data
		end
		qIn = (src[:reinterpolateD](reshape(srcData[1], srcGeometry.nt[1], numTracesSrc), srcGeometry.dt[1], dtComp))'
		dIn = (rec[:reinterpolateD](reshape(recData[1], recGeometry.nt[1], numTracesRec), recGeometry.dt[1], dtComp))'
		ntComp = size(dIn,2)
	end
	ntSrc = Int(trunc(tmaxSrc/dtComp+1))
	ntRec = Int(trunc(tmaxRec/dtComp+1))
	isempty(options.save_rate) ? save_rate=dtComp : save_rate=options.save_rate

	# Set up source coordinates
	src_coords = setup_grid(srcGeometry, model.n, model.o)
	src[:set_receiver_pos](src_coords)
	src[:set_shape](ntComp, numTracesSrc)

	# Set up receiver coordinates
	rec_coords = setup_grid(recGeometry, model.n, model.o)
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
	
	if op=='F'
		if mode==1
			println("Nonlinear forward modeling with density (source no. ",srcnum,")")
			u = zeros(Float32,(dimsfull..., 3))
			dOut = zeros(Float32,rec[:shape])'
			op = Acoustic[:Forward](free_surface=options.free_surface)
			types, args = process_args(op, Dict(:damp=>damp, :m=>m, :rec=>dOut, :rec_coords=>rec_coords', :src=>qIn, :src_coords=>src_coords', :u=>u,
										        :rho=>rho))
			lib = op[:compile]
			status = ccallexpr(lib, Symbol(op[:name]), types, args)
			# zero-padd output if shot record has more samples than source
			if ntRec > ntComp
				dOut = [dOut zeros(size(dOut,1),ntRec - ntComp)]
			end
			dOut = rec[:reinterpolateD](dOut[:,1:ntRec]', dtComp, recGeometry.dt[1])
			if options.save_data_to_disk
				q = joData(srcGeometry,srcData); d = joData(recGeometry,dOut)
				file = join([string(options.file_name),"_",string(srcGeometry.xloc[1][1]),"_",string(srcGeometry.yloc[1][1]),".segy"])
				block_out = joData_to_SeisBlock(d,q)
				segy_write(join([options.file_path,"/",file]), block_out)
				container = scan_file(join([options.file_path,"/",file]),["GroupX","GroupY","dt","SourceSurfaceElevation","RecGroupElevation"])
				return joData(container)
			else
				return joData(recGeometry,dOut)
			end
		elseif mode==-1
			println("Nonlinear adjoint modeling with density (source no. ",srcnum,")")
			v = zeros(Float32,(dimsfull..., 3))
			qOut = zeros(Float32,src[:shape])'
			op = Acoustic[:Adjoint](free_surface=options.free_surface)
			types, args = process_args(op, Dict(:damp=>damp, :m=>m, :rec=>dIn, :rec_coords=>rec_coords', :srca=>qOut, :srca_coords=>src_coords', :v=>v,
										        :rho=>rho))
			lib = op[:compile]
			status = ccallexpr(lib, Symbol(op[:name]), types, args)
			# zero-padd output if source has more samples than data
			if ntSrc > ntComp
				qOut = [qOut zeros(size(qOut,1),ntSrc - ntComp)]
			end
			qOut = src[:reinterpolateD](qOut[:,1:ntSrc]', dtComp, srcGeometry.dt[1])
			return joData(srcGeometry,qOut)
		end
	elseif op=='J'
		if mode==1
			println("Linearized forward modeling with density (source no. ",srcnum,")")
			u = zeros(Float32,(dimsfull..., 3))
			du = zeros(Float32,(dimsfull..., 3))
			dmIn = permutedims(add_pml(reshape(dm, model.n), model.nb), dimsflip)
			dOut = zeros(Float32,rec[:shape])'
			op = Acoustic[:Born](free_surface=options.free_surface)
			types, args = process_args(op, Dict(:damp=>damp, :m=>m, :Lrec=>dOut, :Lrec_coords=>rec_coords', :src=>qIn, :src_coords=>src_coords', :u=>u,
										        :dm=>dmIn, :du=>du, :rho=>rho))
			lib = op[:compile]
			status = ccallexpr(lib, Symbol(op[:name]), types, args)
			# zero-padd output if shot record has more samples than source
			if ntRec > ntComp
				dOut = [dOut zeros(size(dOut,1),ntRec - ntComp)]
			end
			dOut = rec[:reinterpolateD](dOut[:,1:ntRec]', dtComp, recGeometry.dt[1])
			if options.save_data_to_disk
				q = joData(srcGeometry,srcData); d = joData(recGeometry,dOut)
				file = join([string(options.file_name),"_",string(srcGeometry.xloc[1][1]),"_",string(srcGeometry.yloc[1][1]),".segy"])
				block_out = joData_to_SeisBlock(d,q)
				segy_write(join([options.file_path,"/",file]), block_out)
				container = scan_file(join([options.file_path,"/",file]),["GroupX","GroupY","dt","SourceSurfaceElevation","RecGroupElevation"])
				return joData(container)
			else
				return joData(recGeometry,dOut)
			end
		elseif mode==-1
			println("Linearized adjoint modeling with density(source no. ",srcnum,")")
			u = zeros(Float32,(prod(dimsfull), 3))
			nsave = trunc(Int,ntComp/(save_rate/dtComp) +1)
			up = zeros(Float32,(prod(dimsfull),nsave))
			dNull = zeros(Float32,rec[:shape])'
			op = Acoustic[:Forward](save=true, free_surface=options.free_surface)
			types, args = process_args(op, Dict(:damp=>damp, :m=>m, :rec=>dNull, :rec_coords=>rec_coords', :src=>qIn, :src_coords=>src_coords',
										        :u=>u, :usave=>up, :rho=>rho))
			lib = op[:compile]
			status = ccallexpr(lib, Symbol(op[:name]), types, args)
            # gradient
			v = zeros(Float32,(prod(dimsfull), 3))
			grad = zeros(Float32, size(m))
			opg = Acoustic[:Gradient](free_surface=options.free_surface)
			typesg, argsg = process_args(opg, Dict(:damp=>damp, :m=>m, :rec=>dIn, :rec_coords=>rec_coords', :u=>up, :v=>v,
										           :rho=>rho, :grad=>grad))
			libg = opg[:compile]
			status = ccallexpr(libg, Symbol(opg[:name]), typesg, argsg)
			grad = remove_pml(permutedims(reshape(grad,dimsfull),dimsflip),model.nb)
			if options.limit_m==true && model_full.n[3] > 1
				grad = extend_gradient(model_full,model,grad)
			end
			return vec(grad)
		end	
	end
end

