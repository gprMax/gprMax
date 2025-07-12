macros = """
    #define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define INDEX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define INDEX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
    #define IDX2D_MAT(m, n) (m)*($NY_MATCOEFFS)+(n)
    #define IDX2D_MATDISP(m, n) (m)*($NY_MATDISPCOEFFS)+(n)
    #define IDX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS)+(j)*($NZ_FIELDS)+(k)
    #define IDX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID)+(i)*($NY_ID)*($NZ_ID)+(j)*($NZ_ID)+(k)
    #define IDX4D_T(p, i, j, k) (p)*($NX_T)*($NY_T)*($NZ_T)+(i)*($NY_T)*($NZ_T)+(j)*($NZ_T)+(k)
                       
"""