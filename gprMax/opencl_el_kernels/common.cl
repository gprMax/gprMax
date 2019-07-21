#define INDEX2D_RXCOORDS(m,n) (m)*({{NY_RXCOORDS}}) + (n)
#define INDEX3D_RXS(i,j,k) (i)*({{NY_RXS}})*({{NZ_RXS}}) + (j)*({{NZ_RXS}}) + (k)
#define INDEX2D_MATDISP(m, n) (m)*({{NY_MATDISPCOEFFS}}) + (n)
#define INDEX4D_T(p, i, j, k) (p)*({{NX_T}})*({{NY_T}})*({{NZ_T}}) + (i)*({{NY_T}})*({{NZ_T}}) + (j)*({{NZ_T}}) + (k)
#define INDEX2D_MAT(m, n) (m)*({{NY_MATCOEFFS}}) + (n)
#define INDEX2D_SRCINFO(m, n) (m)*({{NY_SRCINFO}}) + (n)
#define INDEX2D_SRCWAVES(m, n) (m)*({{NY_SRCWAVES}}) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*({{NY_FIELDS}})*({{NZ_FIELDS}}) + (j)*({{NZ_FIELDS}}) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*({{NX_ID}})*({{NY_ID}})*({{NZ_ID}}) + (i)*({{NY_ID}})*({{NZ_ID}}) + (j)*({{NZ_ID}}) + (k)
#define INDEX2D_R(m, n) (m)*(NY_R)+(n)
#define INDEX4D_PHI1(p, i, j, k) (p)*(NX_PHI1)*(NY_PHI1)*(NZ_PHI1)+(i)*(NY_PHI1)*(NZ_PHI1)+(j)*(NZ_PHI1)+(k)
#define INDEX4D_PHI2(p, i, j, k) (p)*(NX_PHI2)*(NY_PHI2)*(NZ_PHI2)+(i)*(NY_PHI2)*(NZ_PHI2)+(j)*(NZ_PHI2)+(k)

#include <pyopencl-complex.h>

__constant {{REAL}} updatecoeffsE[{{N_updatecoeffsE}}] = 
{
    {% for i in updateEVal %}
    {{i}},
    {% endfor %}
};

__constant {{REAL}} updatecoeffsH[{{N_updatecoeffsH}}] = 
{
    {% for i in updateHVal %}
    {{i}},
    {% endfor %}
};