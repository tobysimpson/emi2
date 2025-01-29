//
//  mg.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include <math.h>
#include "mg.h"

//init
void mg_ini(struct ocl_obj *ocl, struct mg_obj *mg, struct msh_obj *msh)
{
    //allocate
    mg->lvls = malloc(mg->nl*sizeof(struct lvl_obj));
    
    //levels (skip fine)
    for(int l=1; l<mg->nl; l++)
    {
        struct lvl_obj *lvl = &mg->lvls[l];
        
        //mesh
        lvl->msh.dx = msh->dx*pow(2, l);
        lvl->msh.dt = msh->dt;
        lvl->msh.le = (cl_uint2){msh->le.x-l, msh->le.y-l};
        msh_ini(&lvl->msh);
        
        //memory
        lvl->uu = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);
        lvl->bb = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);
        lvl->rr = clCreateBuffer(ocl->context, CL_MEM_HOST_READ_ONLY, lvl->msh.nv_tot*sizeof(float), NULL, &ocl->err);
    } //lvl

    //kernels
    mg->vtx_zro = clCreateKernel(ocl->program, "vtx_zro", &ocl->err);
    mg->vtx_prj = clCreateKernel(ocl->program, "vtx_prj", &ocl->err);
    mg->vtx_itp = clCreateKernel(ocl->program, "vtx_itp", &ocl->err);
    
    mg->ops[0].vtx_rsd = clCreateKernel(ocl->program, "vtx_rsd0", &ocl->err);
    mg->ops[0].vtx_jac = clCreateKernel(ocl->program, "vtx_jac0", &ocl->err);
    
    mg->ops[1].vtx_rsd = clCreateKernel(ocl->program, "vtx_rsd1", &ocl->err);
    mg->ops[1].vtx_jac = clCreateKernel(ocl->program, "vtx_jac1", &ocl->err);
    
    mg->ops[2].vtx_rsd = clCreateKernel(ocl->program, "vtx_rsd2", &ocl->err);
    mg->ops[2].vtx_jac = clCreateKernel(ocl->program, "vtx_jac2", &ocl->err);
    
    return;
}


//solve
void mg_slv(struct ocl_obj *ocl, struct mg_obj *mg, struct op_obj *op)
{
    //cycle
    for(int c=0; c<mg->nc; c++)
    {
//        printf("cyc %2d\n", c);
        
        //descend
        for(int l=0; l<mg->nl; l++)
        {
//            printf("lvl %2d\n", l);
            
            //skip top
            if(l>0)
            {
                //project
                ocl->err = clSetKernelArg(mg->vtx_prj,  0, sizeof(struct msh_obj),    (void*)&mg->lvls[l].msh);    //coarse
                ocl->err = clSetKernelArg(mg->vtx_prj,  1, sizeof(cl_mem),            (void*)&mg->lvls[l].uu);     //coarse
                ocl->err = clSetKernelArg(mg->vtx_prj,  2, sizeof(cl_mem),            (void*)&mg->lvls[l].bb);     //coarse
                ocl->err = clSetKernelArg(mg->vtx_prj,  3, sizeof(cl_mem),            (void*)&mg->lvls[l-1].rr);   //fine
                
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vtx_prj, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
            }
            
            //jacobi iter
            for(int j=0; j<mg->nj; j++)
            {
                //residual
                ocl->err = clSetKernelArg(op->vtx_rsd,  0, sizeof(struct msh_obj),    (void*)&mg->lvls[l].msh);
                ocl->err = clSetKernelArg(op->vtx_rsd,  1, sizeof(cl_mem),            (void*)&mg->lvls[l].uu);
                ocl->err = clSetKernelArg(op->vtx_rsd,  2, sizeof(cl_mem),            (void*)&mg->lvls[l].bb);
                ocl->err = clSetKernelArg(op->vtx_rsd,  3, sizeof(cl_mem),            (void*)&mg->lvls[l].rr);
                
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_rsd, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
                
                //jacobi
                ocl->err = clSetKernelArg(op->vtx_jac,  0, sizeof(struct msh_obj),    (void*)&mg->lvls[l].msh);
                ocl->err = clSetKernelArg(op->vtx_jac,  1, sizeof(cl_mem),            (void*)&mg->lvls[l].uu);
                ocl->err = clSetKernelArg(op->vtx_jac,  2, sizeof(cl_mem),            (void*)&mg->lvls[l].rr);
                
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_jac, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_rsd, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
            
        } //l
        
        
        //ascend
        for(int l=(mg->nl-2); l>=0; l--)
        {
//            printf("lvl %d\n", l);
            
            //interp
            ocl->err = clSetKernelArg(mg->vtx_itp,  0, sizeof(struct msh_obj),    (void*)&mg->lvls[l].msh);    //fine
            ocl->err = clSetKernelArg(mg->vtx_itp,  1, sizeof(cl_mem),            (void*)&mg->lvls[l+1].uu);   //coarse
            ocl->err = clSetKernelArg(mg->vtx_itp,  2, sizeof(cl_mem),            (void*)&mg->lvls[l].uu);     //fine
            
            ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, mg->vtx_itp, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
            
            //jacobi iter
            for(int j=0; j<mg->nj; j++)
            {
                //residual
                ocl->err = clSetKernelArg(op->vtx_rsd,  0, sizeof(struct msh_obj),    (void*)&mg->lvls[l].msh);
                ocl->err = clSetKernelArg(op->vtx_rsd,  1, sizeof(cl_mem),            (void*)&mg->lvls[l].uu);
                ocl->err = clSetKernelArg(op->vtx_rsd,  2, sizeof(cl_mem),            (void*)&mg->lvls[l].bb);
                ocl->err = clSetKernelArg(op->vtx_rsd,  3, sizeof(cl_mem),            (void*)&mg->lvls[l].rr);
                
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_rsd, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
                
                //jacobi
                ocl->err = clSetKernelArg(op->vtx_jac,  0, sizeof(struct msh_obj),    (void*)&mg->lvls[l].msh);
                ocl->err = clSetKernelArg(op->vtx_jac,  1, sizeof(cl_mem),            (void*)&mg->lvls[l].uu);
                ocl->err = clSetKernelArg(op->vtx_jac,  2, sizeof(cl_mem),            (void*)&mg->lvls[l].rr);
                
                ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_jac, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
            }
            
            //residual
            ocl->err = clEnqueueNDRangeKernel(ocl->command_queue, op->vtx_rsd, 2, NULL, (size_t*)&mg->lvls[l].msh.nv, NULL, 0, NULL, NULL);
            
        } //l
    } //c
}





//final
void mg_fin(struct ocl_obj *ocl, struct mg_obj *mg)
{
    //kernels
    ocl->err = clReleaseKernel(mg->vtx_prj);
    ocl->err = clReleaseKernel(mg->vtx_itp);
    
    ocl->err = clReleaseKernel(mg->ops[0].vtx_rsd);
    ocl->err = clReleaseKernel(mg->ops[0].vtx_jac);
    
    ocl->err = clReleaseKernel(mg->ops[1].vtx_rsd);
    ocl->err = clReleaseKernel(mg->ops[1].vtx_jac);
    
    ocl->err = clReleaseKernel(mg->ops[2].vtx_rsd);
    ocl->err = clReleaseKernel(mg->ops[2].vtx_jac);
    
    //levels
    for(int l=1; l<mg->nl; l++)
    {
        //memory
        ocl->err = clReleaseMemObject(mg->lvls[l].uu);
        ocl->err = clReleaseMemObject(mg->lvls[l].bb);
        ocl->err = clReleaseMemObject(mg->lvls[l].rr);
    }
    
    //deallocate
    free(mg->lvls);
    
    return;
}
