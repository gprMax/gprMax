from string import Template

macros = Template("""
    #define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX2D_R(m, n) (m)*(NY_R)+(n)
    #define IDX4D_PHI1(p, i, j, k) (p)*(NX_PHI1)*(NY_PHI1)*(NZ_PHI1)+(i)*(NY_PHI1)*(NZ_PHI1)+(j)*(NZ_PHI1)+(k)
    #define IDX4D_PHI2(p, i, j, k) (p)*(NX_PHI2)*(NY_PHI2)*(NZ_PHI2)+(i)*(NY_PHI2)*(NZ_PHI2)+(j)*(NZ_PHI2)+(k)         
""")