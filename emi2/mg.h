//
//  mg.h
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef mg_h
#define mg_h

#include "msh.h"


//level
struct lvl_obj
{
    //mesh
    struct msh_obj  msh;
    
    //memory
    cl_mem          uu;
    cl_mem          bb;
    cl_mem          rr;
};


//params
struct mg_obj
{
    cl_uint         nl;     //depth
    cl_uint         nc;     //num cycles
    cl_uint         nj;     //jac iter
    
    //levels
    struct lvl_obj  *lvls;
    
    //trans ops
    cl_kernel       vtx_zro;    //reset
    cl_kernel       vtx_prj;    //project
    cl_kernel       vtx_itp;    //interp
    
    //diff ops
    struct op_obj   ops[3];     //rsd,jac
};



//methods
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg, struct msh_obj *msh);
void mg_slv(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op);
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg);


#endif /* mg_h */
