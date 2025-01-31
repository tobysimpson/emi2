//
//  ocl.h
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef ocl_h
#define ocl_h

#include <stdio.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


//operator kernels
struct op_obj
{
    cl_kernel   vtx_rsd;        //residual
    cl_kernel   vtx_jac;        //jacobi
};


//object
struct ocl_obj
{
    //environment
    cl_int              err;
    cl_platform_id      platform_id;
    cl_device_id        device_id;
    cl_uint             num_devices;
    cl_uint             num_platforms;
    cl_context          context;
    cl_command_queue    command_queue;
    cl_program          program;
    cl_event            event;
    
    //debug
    char                device_char[50];
    cl_uint             device_uint;
    
    //kernels
    cl_kernel           vtx_ini;    //init
    cl_kernel           vtx_tst;    //membrane
    
};



//functions
void ocl_ini(struct ocl_obj *ocl);
void ocl_fin(struct ocl_obj *ocl);

#endif /* ocl_h */
