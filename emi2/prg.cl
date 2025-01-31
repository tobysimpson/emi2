//
//  prg.cl
//  mg2
//
//  Created by Toby Simpson on 05.12.2024.
//  Copyright © 2024 Toby Simpson. All rights reserved.
//


/*
 ===================================
 struct
 ===================================
 */

//object
struct msh_obj
{
    float    dt;
    float    dx;
    
    uint2    le;
    ulong2   ne;
    ulong2   nv;
    ulong2   iv;
    
    ulong    ne_tot;
    ulong    nv_tot;
    
    float    dx2;    //dx*dx
    float    rdx2;   //1/(dx*dx)
    long2    ne2;    //ne/2
};

/*
 ===================================
 const
 ===================================
 */

//stencil
constant ulong2 off_fac[4] = {{-1,0},{+1,0},{0,-1},{0,+1}};

//conductivity
constant float MD_SIG_H = 1.0f;        //conductivity (mS mm^-1) = muA mV^-1 mm^-1
constant float MD_SIG_T = 1.0f;

/*
 ===================================
 util
 ===================================
 */

//global index
ulong fn_idx1(ulong2 pos, ulong2 dim)
{
    return pos.x + dim.x*pos.y;
}

//in-bounds
int fn_bnd1(ulong2 pos, ulong2 dim)
{
    return all(pos>=0)*all(pos<dim);
}

//on the boundary
int fn_bnd2(ulong2 pos, ulong2 dim)
{
    return any(pos==0)||any(pos==dim-1);    //not tested
    
//    return (pos.x==0)||(pos.y==0)||(pos.z==0)||(pos.x==dim.x-1)||(pos.y==dim.y-1)||(pos.z==dim.z-1);
}

//coordinate - !adj for stupid test!
float2 fn_x1(ulong2 pos, const struct msh_obj *msh)
{
    return msh->dx*convert_float2(convert_long2(pos) - msh->ne2);
}

/*
 ===================================
 data
 ===================================
 */

//solution
float fn_u1(float2 x)
{
    return sin(x.x);
}

//rhs
float fn_b1(float2 x)
{
    return  -sin(x.x);
}

/*
 ===================================
 sdf
 ===================================
 */


//cuboid
float sdf_cub(float2 x, float2 c, float2 r)
{
    float2 d = fabs(x - c) - r;
    
    return max(d.x, d.y);
}

//sphere
float sdf_sph(float2 x, float2 c, float r)
{
    return length(x - c) - r;
}

/*
 ===================================
 sets
 ===================================
 */

//stimulus
float fn_g0(float2 x)
{
    float2 c = (float2){0.25f,0.25f};
    float2 r = (float2){0.25f,0.25f};

    return sdf_cub(x, c, r);
}

//geometry
float fn_g1(float2 x)
{
    float2 c = (float2){0.00f,0.00f};
    float2 r = (float2){0.25f,0.25f};

    return sdf_cub(x, c, r);
}

//geometry
float fn_g2(float2 x)
{
//    x = remainder(x + 0.25f , 0.5f);
    
    int2 q;
    
    float2 r = remquo(x, 0.5f, &q);
    
    return r.x;
    
//    float2 c = (float2){0.00f,0.00f};
////    float2 r = (float2){0.2f,0.2f};
//
//    return sdf_sph(x, c, 0.25f);
}


/*
 ===================================
 multigrid
 ===================================
 */


//reset
kernel void vtx_zro(const struct msh_obj    msh,
                    global float            *uu)
{
    ulong2 vtx_pos  = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx  = fn_idx1(vtx_pos, msh.nv);

    uu[vtx_idx] = 0e0f;

    return;
}


//projection
kernel void vtx_prj(const  struct msh_obj   msh,    //coarse    (out)
                    global float            *uu,    //coarse    (out reset)
                    global float            *bb,    //coarse    (out)
                    global float            *rr)    //fine      (in)
{
    ulong2 vtx_pos   = {get_global_id(0), get_global_id(1)}; //coarse
    ulong  vtx_idx0  = fn_idx1(vtx_pos, msh.nv);
    
    //injection
    ulong  vtx_idx1  = fn_idx1(2*vtx_pos, 2*msh.ne+1);
    
    //store
    uu[vtx_idx0] = 0e0f;
    bb[vtx_idx0] = rr[vtx_idx1];

    return;
}


//interpolation
kernel void vtx_itp(const  struct msh_obj   msh,    //fine      (out)
                    global float            *u0,    //coarse    (in)
                    global float            *u1)    //fine      (out)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)}; //fine
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);   //fine
    
    //coarse
    float2 pos = convert_float2(vtx_pos)/2e0f;
    
    //round up/down
    ulong2 pos0 = convert_ulong2(floor(pos));
    ulong2 pos1 = convert_ulong2(ceil(pos));
    
    ulong2 dim = 1+(msh.nv-1)/2;
    
    float s = 0e0f;
    s += u0[fn_idx1((ulong2){pos0.x, pos0.y}, dim)];
    s += u0[fn_idx1((ulong2){pos1.x, pos0.y}, dim)];
    s += u0[fn_idx1((ulong2){pos0.x, pos1.y}, dim)];
    s += u0[fn_idx1((ulong2){pos1.x, pos1.y}, dim)];

    
    u1[vtx_idx] += s/4e0f;
    
    return;
}




/*
 ===================================
 kernels
 ===================================
 */

//init
kernel void vtx_ini(const struct msh_obj    msh,
                    global float            *uu,
                    global float            *bb,
                    global float            *rr,
                    global float            *vv,
                    global float            *ww,
                    global float            *gg)
{
    ulong2 vtx_pos  = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float2 x = fn_x1(vtx_pos, &msh);
    
//    printf("%3lu %v3lu\n", vtx_idx, vtx_pos);
    
    float u = 8.0f*M_PI_F*M_PI_F*sin(2.0f*M_PI_F*x.x)*sin(2.0f*M_PI_F*x.y);
    
    uu[vtx_idx] = 0.0f;
    bb[vtx_idx] = u;
    rr[vtx_idx] = 0.0f;
    vv[vtx_idx] = 0.0f;
    ww[vtx_idx] = 0.0f;
    gg[vtx_idx] = fn_g1(x);
    
    return;
}


//init
kernel void vtx_tst(int                     t,
                    const struct msh_obj    msh,
                    global float            *uu,
                    global float            *vv,
                    global float            *ww)
{
    ulong2 vtx_pos  = (ulong2){get_global_id(0) + 1, get_global_id(1) + 1}; //interior
    ulong  vtx_idx  = fn_idx1(vtx_pos, msh.nv);
    
    float2 x = fn_x1(vtx_pos, &msh);
    
    float  s = 0.0f;
    
    //stencil
    for(int k=0; k<4; k++)
    {
//        ulong2  adj_pos = vtx_pos + off_fac[k];
//        ulong   adj_idx = fn_idx1(adj_pos, msh.nv);

        s += fn_g1(x)<=0.0f;
    }
    
    uu[vtx_idx] = s;
    vv[vtx_idx] = t;
    
    return;
}


/*
 ====================
 poisson1
 ====================
 */


//poisson Au = b


//residual
kernel void vtx_rsd0(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float            *rr)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float2 x = fn_x1(vtx_pos, &msh);
    
    float  s = 0.0f;    //off-diag
    float  d = 0.0f;    //diag
    
    //stencil
    for(int k=0; k<4; k++)
    {
        ulong2  adj_pos = vtx_pos + off_fac[k];
        ulong   adj_idx = fn_idx1(adj_pos, msh.nv);
        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //domain

        d += uu[vtx_idx];                               //zero dirichlet
        
        //domain
        if(adj_bnd)
        {
//            d += uu[vtx_idx];                         //zero neumann
            s += uu[adj_idx];
        }
    }
    
    //operator
    float Au = msh.rdx2*(s - d);
    
    //residual
    rr[vtx_idx] = (fn_g1(x)>0.0f)?bb[vtx_idx] - Au:0.0f;    //geometry
//    rr[vtx_idx] = bb[vtx_idx] - Au;                         //no geometry

    return;
}


//jacobi
kernel void vtx_jac0(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *rr)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float  d = 0.0f;    //degree
    
    //stencil
    for(int k=0; k<4; k++)
    {
//        ulong2  adj_pos = vtx_pos + off_fac[k];
//        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //domain
        
        d += 1e0f;                      //zero dirichlet
//        d += adj_bnd;                 //zero neumann
    }
    
    //du = D^-1(r)
    uu[vtx_idx] += (msh.dx2)*rr[vtx_idx]/-d;

    return;
}


/*
 ====================
 poisson2
 ====================
 */


//poisson Au = b


//residual
kernel void vtx_rsd1(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float            *rr)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float2 x = fn_x1(vtx_pos, &msh);
    
    float  s = 0.0f;    //off-diag
    float  d = 0.0f;    //diag
    
    //stencil
    for(int k=0; k<4; k++)
    {
        ulong2  adj_pos = vtx_pos + off_fac[k];
        ulong   adj_idx = fn_idx1(adj_pos, msh.nv);
        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //domain

        d += uu[vtx_idx];                               //zero dirichlet
        
        //domain
        if(adj_bnd)
        {
//            d += uu[vtx_idx];                         //zero neumann
            s += uu[adj_idx];
        }
    }
    
    //operator
    float Au = msh.rdx2*(s - d);
    
    //residual
    rr[vtx_idx] = (fn_g1(x)<0.0f)?bb[vtx_idx] - Au:0.0f;    //geometry
//    rr[vtx_idx] = bb[vtx_idx] - Au;                         //no geometry

    return;
}


//jacobi
kernel void vtx_jac1(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *rr)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float  d = 0.0f;    //degree
    
    //stencil
    for(int k=0; k<4; k++)
    {
//        ulong2  adj_pos = vtx_pos + off_fac[k];
//        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //domain
        
        d += 1e0f;                      //zero dirichlet
//        d += adj_bnd;                 //zero neumann
    }
    
    //du = D^-1(r)
    uu[vtx_idx] += (msh.dx2)*rr[vtx_idx]/-d;

    return;
}



/*
 ====================
 implicit euler
 ====================
 */


//implicit euler (I - alp*A)uˆ(t+1) = uˆ(t)


//residual
kernel void vtx_rsd2(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *bb,
                     global float            *rr)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float2 x = fn_x1(vtx_pos, &msh);
    
    float  s = 0.0f;    //L+U
    float  d = 0.0f;    //D
    
    //stencil
    for(int k=0; k<4; k++)
    {
        ulong2  adj_pos = vtx_pos + off_fac[k];
        ulong   adj_idx = fn_idx1(adj_pos, msh.nv);
        int     adj_bnd = fn_bnd1(adj_pos, msh.nv);     //domain

//        d += uu[vtx_idx];                             //zero dirichlet
        
        //domain
        if(adj_bnd)
        {
            d += uu[vtx_idx];                           //zero neumann
            s += uu[adj_idx];
        }
    }
    //constants
    float alp = MD_SIG_H*msh.dt*msh.rdx2;
    
    //operator (I-alp*A)*u
    float Au = uu[vtx_idx] - alp*(s - d);
    
    //residual
    rr[vtx_idx] = (fn_g1(x)<=0e0f)?bb[vtx_idx] - Au:0e0f;    //geom
//    rr[vtx_idx] = bb[vtx_idx] - Au;                       //no geom

    return;
}


//jacobi
kernel void vtx_jac2(const  struct msh_obj   msh,
                     global float            *uu,
                     global float            *rr)
{
    ulong2 vtx_pos = {get_global_id(0), get_global_id(1)};
    ulong  vtx_idx = fn_idx1(vtx_pos, msh.nv);
    
    float  d = 0.0f;    //degree
    
    //stencil
    for(int k=0; k<4; k++)
    {
        ulong2  adj_pos = vtx_pos + off_fac[k];
        
//        d += 1e0f;                    //zero dirichlet
        
        //domain
        if(fn_bnd1(adj_pos, msh.nv))
        {
            d += 1e0f;                    //zero neumann
        }
    }
    //constants
    float alp = MD_SIG_H*msh.dt*msh.rdx2;
    
    //du = D^-1(r)
    uu[vtx_idx] += rr[vtx_idx]/(1e0f + alp*d);

    return;
}
