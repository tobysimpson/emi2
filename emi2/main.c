//
//  main.c
//  emi1
//
//  Created by Toby Simpson on 20.01.2025.
//  Copyright © 2025 Toby Simpson. All rights reserved.
//

#include <stdio.h>
#include <math.h>

#include "mg.h"
#include "io.h"


//trying emi with edge-based membrane
int main(int argc, const char * argv[])
{
    printf("hello\n");
    
//    printf("%lu %lu\n", sizeof(unsigned char), sizeof(char));
    
    //ocl
    struct ocl_obj ocl;
    ocl_ini(&ocl);
    
    //size
    int n  = 4;
    
    //mesh (fine)
    struct msh_obj msh;
    msh.dt = 0.01f;
    msh.dx = 1.0f/pow(2.0f,n);
    msh.le = (cl_uint2){n,n};
    msh_ini(&msh);
    
    //mg
    struct mg_obj mg;
    mg.nl = 2;
    mg.nj = 3;
    mg.nc = 1;
    mg_ini(&ocl, &mg, &msh);
    

    //memory
    cl_mem uu = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(float), NULL, &ocl.err);
    cl_mem bb = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(float), NULL, &ocl.err);
    cl_mem rr = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(float), NULL, &ocl.err);
    cl_mem vv = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(float), NULL, &ocl.err);
    cl_mem ww = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(float), NULL, &ocl.err);
    cl_mem gg = clCreateBuffer(ocl.context, CL_MEM_HOST_READ_ONLY, msh.nv_tot*sizeof(float), NULL, &ocl.err);
    
    /*
     =============================
     static args
     =============================
     */
    
    //init
    ocl.err = clSetKernelArg(ocl.vtx_ini,  0, sizeof(struct msh_obj),    (void*)&msh);
    ocl.err = clSetKernelArg(ocl.vtx_ini,  1, sizeof(cl_mem),            (void*)&uu);
    ocl.err = clSetKernelArg(ocl.vtx_ini,  2, sizeof(cl_mem),            (void*)&bb);
    ocl.err = clSetKernelArg(ocl.vtx_ini,  3, sizeof(cl_mem),            (void*)&rr);
    ocl.err = clSetKernelArg(ocl.vtx_ini,  4, sizeof(cl_mem),            (void*)&vv);
    ocl.err = clSetKernelArg(ocl.vtx_ini,  5, sizeof(cl_mem),            (void*)&ww);
    ocl.err = clSetKernelArg(ocl.vtx_ini,  6, sizeof(cl_mem),            (void*)&gg);
    
    //test - 0 is t
    ocl.err = clSetKernelArg(ocl.vtx_tst,  1, sizeof(struct msh_obj),    (void*)&msh);
    ocl.err = clSetKernelArg(ocl.vtx_tst,  2, sizeof(cl_mem),            (void*)&uu);
    ocl.err = clSetKernelArg(ocl.vtx_tst,  3, sizeof(cl_mem),            (void*)&vv);
    ocl.err = clSetKernelArg(ocl.vtx_tst,  4, sizeof(cl_mem),            (void*)&ww);
    
    
    /*
     =============================
     run
     =============================
     */
    
    //ini
    ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_ini, 2, NULL, (size_t*)&msh.nv, NULL, 0, NULL, NULL);
    
    //time
    int t = 0;

    //frames
    for(int frm=0; frm<100; frm++)
    {
        if(frm%10==0)
        {
            printf("frm %2d %04d\n", frm, t);
        }
        
        //write
        wrt_xmf(&ocl, &msh, frm);
        wrt_raw(&ocl, &msh, &uu, "uu", frm);
        wrt_raw(&ocl, &msh, &bb, "bb", frm);
        wrt_raw(&ocl, &msh, &rr, "rr", frm);
        wrt_raw(&ocl, &msh, &vv, "vv", frm);
        wrt_raw(&ocl, &msh, &ww, "ww", frm);
        wrt_raw(&ocl, &msh, &gg, "gg", frm);
        
        //timestep
        for(int itr=0; itr<10; itr++)
        {
            //test
            ocl.err = clSetKernelArg(ocl.vtx_tst,  0, sizeof(int), (void*)&t);
            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_tst, 2, NULL, (size_t*)&msh.iv, NULL, 0, NULL, NULL);
            
            /*
             =============================
             poisson1
             =============================
             */
            
//            //set fine
//            mg.lvls[0].msh = msh;
//            mg.lvls[0].uu  = uu;
//            mg.lvls[0].bb  = bb;
//            mg.lvls[0].rr  = rr;
            
//            //reset
//            ocl.err = clSetKernelArg(ocl.vtx_zro,  0, sizeof(struct msh_obj),    (void*)&msh);
//            ocl.err = clSetKernelArg(ocl.vtx_zro,  1, sizeof(cl_mem),            (void*)&bb);
//            ocl.err = clEnqueueNDRangeKernel(ocl.command_queue, ocl.vtx_zro, 2, NULL, (size_t*)&msh.nv, NULL, 0, NULL, NULL);
            
            //poisson0
//            mg_slv(&ocl, &mg, &mg.ops[0]);

        } //t

    } //f
    
    //flush
    ocl.err = clFlush(ocl.command_queue);
    ocl.err = clFinish(ocl.command_queue);
    
    //memory
    ocl.err = clReleaseMemObject(uu);
    ocl.err = clReleaseMemObject(bb);
    ocl.err = clReleaseMemObject(rr);
    ocl.err = clReleaseMemObject(vv);
    ocl.err = clReleaseMemObject(ww);
    ocl.err = clReleaseMemObject(gg);
    
    //final
    mg_fin(&ocl, &mg);
    ocl_fin(&ocl);
    
    printf("done\n");
    
    return 0;
}

