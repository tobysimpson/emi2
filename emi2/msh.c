//
//  msh.c
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#include "msh.h"


//init
void msh_ini(struct msh_obj *msh)
{
    msh->ne     = (cl_ulong2){1<<msh->le.x, 1<<msh->le.y};
    msh->nv     = (cl_ulong2){msh->ne.x+1, msh->ne.y+1};
    
    msh->ne_tot = msh->ne.x*msh->ne.y;
    msh->nv_tot = msh->nv.x*msh->nv.y;
    
    msh->dx2    = msh->dx*msh->dx;
    msh->rdx2   = 1e0f/msh->dx2;
    msh->ne2    = (cl_long2){msh->ne.x/2, msh->ne.y/2};
    
    printf("msh %f %f %02d%02d [%3llu %3llu] %10llu\n", msh->dt, msh->dx, msh->le.x, msh->le.y, msh->ne.x, msh->ne.y, msh->nv_tot);
    
    return;
}
