//
//  io.h
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//

#ifndef io_h
#define io_h


#include "mg.h"

#define ROOT_WRITE  "/Users/toby/Downloads"

void wrt_xmf(struct ocl_obj *ocl, struct msh_obj *msh, int idx);
void wrt_raw(struct ocl_obj *ocl, struct msh_obj *msh, cl_mem *buf, char *dsc, int idx);


#endif /* io_h */
