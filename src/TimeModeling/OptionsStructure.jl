# Options structure 
# Author: Philipp Witte, pwitte@eos.ubc.ca
# Date: May 2017
#

export Options

# Object for velocity/slowness models
type Options
	space_order::Integer
	retry_n::Integer
	limit_m::Bool
	buffer_size::Real
	save_rate::Union{Real,Array{Any,1}}
	save_data_to_disk::Bool
	file_path::String
	file_name::String
	free_surface::Bool
end

"""
    Options
		space_order::Integer
        retry_n::Integer
        limit_m::Bool
        buffer_size::Real
        save_rate::Real
        save_data_to_disk::Bool
        file_path::String
        file_name::String
        free_surface::Bool

Options structure for seismic modeling.

`space_order`: order of finite difference stencils of discretized wave equations (needs to be a multiple of 4)

`retry_n`: retry modeling operations in case of worker failure up to `retry_n` times

`limit_m`: for 3D modeling, limit modeling domain to area with receivers (saves memory)

`buffer_size`: if `limit_m=true`, define buffer area on each side of modeling domain (in meters)

`save_rate`: for gradients, save background wavefield at this rate (in milliseconds)

`save_data_to_disk`: if `true`, saves shot records as separate SEG-Y files

`file_path`: path to directory where data is saved

`file_name`: shot records will be saved as specified file name plus its source coordinates

`free_surface`: use free surface if `true`

Constructor
==========

All arguments are optional keyword arguments with the following default values:

    Options(;space_order=8, retry_n=0, limit_m=false, buffer_size=1e3, save_rate=4f0, save_data_to_disk=false,
	         file_path=pwd(), file_name="shot", free_surface=false)

"""
Options(;space_order=8,retry_n=0,limit_m=false,buffer_size=1e3, save_rate=[], save_data_to_disk=false, file_path="", file_name="shot", free_surface=false) = 
	Options(space_order,retry_n,limit_m,buffer_size,save_rate,save_data_to_disk,file_path,file_name, free_surface)


