from string import Template

kernel_template_os = Template("""
// Defining Macros
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS) + (j)*($NZ_FIELDS) + (k)
#define INDEX3D_SUBFIELDS(i, j, k) (i)*($NY_SUBFIELDS)*($NZ_SUBFIELDS) + (j)*($NZ_SUBFIELDS) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID) + (i)*($NY_ID)*($NZ_ID) + (j)*($NZ_ID) + (k)


/////////////////////////
// Electric OS updates //
/////////////////////////


__global__ void hsg_update_electric_os(
    const int face,
    const unsigned int co,
    const int sign_n, const int sign_f,
    const int mid,
    const int sub_ratio,
    const int surface_sep,
    const int n_boundary_cells,
    const int nwn,
    const unsigned int lookup_id,
    const int l_l, const int l_u,
    const int m_l, const int m_u,
    const int n_l, const int n_u,
    $REAL* updatecoeffsE,
    const unsigned int* ID,
    $REAL* field,
    $REAL* inc_field) {
        
        //  This function updates the electric fields of OS.
        // 
        //  Args: 
        //      face: Determines which face of the cube to calculate for.
        //      co: Coefficient used by gprMax update equations which
        //      is specific to the field component being updated.
        //      sign_n, sign_f: Sign of the incident field on the near and far face respectively.
        //      mid: Checks if the H node is midway along the lower edge.
        //      sub_ratio: The ratio of sub-gridding.
        //      surface_sep: The separation between IS and OS.
        //      n_boundary_cells: the number of boundary cells associated with the grid.
        //      nwn: Value of the field that is not changing for each 2D slice.
        //      lookup_id: The index value of the component of the main_grid fields.
        //      l_l, l_u, m_l, m_u, n_l, n_u: The upper and lower limits of the passed fields.
        //      updatecoeffsH: The update coefficients of Magnetic field.
        //      ID: The ID of the components.
        //      field: The main_grid field to be updated.
        //      inc_field: The sub_grid field involved in updating the main_grid field.


        // Current Thread Index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int l, m, l_s, m_s, n_s_l, n_s_r, i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;
        int os;
        double inc_n, inc_f;
        
        // Surface normal index for the subgrid near face h nodes (left i index)
        n_s_l = n_boundary_cells - (surface_sep * sub_ratio) - sub_ratio + floor((double) sub_ratio / 2);

        // Surface normal index for the subgrid far face h nodes (right i index)
        n_s_r = n_boundary_cells + nwn + (surface_sep * sub_ratio) + floor((double) sub_ratio / 2);

        // OS at the left face
        os = n_boundary_cells - (sub_ratio * surface_sep);

        // Linear Index to subscript
        // Since we are only concerned for a 2D slice of the grid with no
        // circular wrapping, the calculation done below works 
        l = idx / ($NZ_FIELDS * $NY_FIELDS); 
        m = idx % ($NZ_FIELDS * $NY_FIELDS);

        // If block for front and back face calculation
        if(face == 3 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // Subgrid coordinates
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = n_l; k0 = m;
            
            // Sub-grid Index
            i1 = l_s; j1 = n_s_l; k1 = m_s;
            i2 = l; j2 = n_u; k2 = m;
            i3 = l_s; j3 = n_s_r; k3 = m_s;

            // Getting the material at the main grid index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];

            // Associated incident field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;

            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }

        // If block for left and right face calculation
        if(face == 2 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = n_l; j0 = l; k0 = m;

            // Sub-grid Index
            i1 = n_s_l; j1 = l_s; k1 = m_s;
            i2 = n_u; j2 = l; k2 = m;
            i3 = n_s_r; j3 = l_s; k3 = m_s;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }

        // If block for top and bottom face calculation
        if(face == 1 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = m; k0 = n_l;

            // Sub-grid Index
            i1 = l_s; j1 = m_s; k1 = n_s_l;
            i2 = l; j2 = m; k2 = n_u;
            i3 = l_s; j3 = m_s; k3 = n_s_r;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsE[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsE[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }
    }


/////////////////////////
// Magnetic OS updates //
/////////////////////////

__global__ void hsg_update_magnetic_os(
    const int face,
    const unsigned int co,
    const int sign_n, const int sign_f,
    const int mid,
    const int sub_ratio,
    const int surface_sep,
    const int n_boundary_cells,
    const int nwn,
    const unsigned int lookup_id,
    const int l_l, const int l_u,
    const int m_l, const int m_u,
    const int n_l, const int n_u,
    $REAL* updatecoeffsH,
    const unsigned int* ID,
    $REAL* field,
    $REAL* inc_field) {
        // Current Thread Index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        
        int l, m, l_s, m_s, n_s_l, n_s_r, i0, j0, k0, i1, j1, k1, i2, j2, k2, i3, j3, k3;
        int os;
        double inc_n, inc_f;

        // Index (normal to os) for the subgrid near face e node
        n_s_l = n_boundary_cells - sub_ratio * surface_sep;

        // Normal index for the subgrid far face e node
        n_s_r = n_boundary_cells + nwn + sub_ratio * surface_sep;

        // OS Inner index for the subgrid
        os = n_boundary_cells - sub_ratio * surface_sep;

        // Linear Index to subscript 
        l = idx / ($NZ_FIELDS * $NY_FIELDS); 
        m = idx % ($NZ_FIELDS * $NY_FIELDS);

        if(face == 3 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords for front face
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                // subgrid coords for back face
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = n_l; k0 = m;
            
            // Sub-grid Index
            i1 = l_s; j1 = n_s_l; k1 = m_s;
            i2 = l; j2 = n_u; k2 = m;
            i3 = l_s; j3 = n_s_r; k3 = m_s;
            
            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsH[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsH[INDEX2D_MAT(material_e_r, co)] * inc_f;               
        }

        if(face == 2 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = n_l; j0 = l; k0 = m;

            // Sub-grid Index
            i1 = n_s_l; j1 = l_s; k1 = m_s;
            i2 = n_u; j2 = l; k2 = m;
            i3 = n_s_r; j3 = l_s; k3 = m_s;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsH[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsH[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }

        if(face == 1 && l >= l_l && l < l_u && m >= m_l && m < m_u) {
            if(mid == 1) {
                // subgrid coords
                l_s = os + (l - l_l) * sub_ratio + floor((double) sub_ratio / 2);
                m_s = os + (m - m_l) * sub_ratio;   
            }
            else {
                l_s = os + (l - l_l) * sub_ratio;
                m_s = os + (m - m_l) * sub_ratio + floor((double) sub_ratio / 2);
            }

            // Main grid Index
            i0 = l; j0 = m; k0 = n_l;

            // Sub-grid Index
            i1 = l_s; j1 = m_s; k1 = n_s_l;
            i2 = l; j2 = m; k2 = n_u;
            i3 = l_s; j3 = m_s; k3 = n_s_r;

            // Material at main grid Index
            int material_e_l = ID[INDEX4D_ID(lookup_id, i0, j0, k0)];
            
            // Associated Incident Field
            inc_n = inc_field[INDEX3D_SUBFIELDS(i1, j1, k1)] * sign_n;
            field[INDEX3D_FIELDS(i0, j0, k0)] += updatecoeffsH[INDEX2D_MAT(material_e_l, co)] * inc_n;
            
            int material_e_r = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_f = inc_field[INDEX3D_SUBFIELDS(i3, j3, k3)] * sign_f;

            field[INDEX3D_FIELDS(i2, j2, k2)] += updatecoeffsH[INDEX2D_MAT(material_e_r, co)] * inc_f;
        }
    }
""")

kernel_template_is = Template("""
// Defining Macros
#define INDEX2D_MAT(m, n) (m)*($NY_MATCOEFFS) + (n)
#define INDEX3D_FIELDS(i, j, k) (i)*($NY_FIELDS)*($NZ_FIELDS) + (j)*($NZ_FIELDS) + (k)
#define INDEX4D_ID(p, i, j, k) (p)*($NX_ID)*($NY_ID)*($NZ_ID) + (i)*($NY_ID)*($NZ_ID) + (j)*($NZ_ID) + (k)

__global__ void hsg_update_is(
    const int nwx, const int nwy, const int nwz,
    const int n,
    const int offset,
    const int nwl, const int nwm,
    const int face,
    const int co,
    const int sign_l, const int sign_u,
    const unsigned int lookup_id,
    const int pre_coeff,
    $REAL* updatecoeffs,
    const unsigned int* ID,
    $REAL* field,
    $REAL* inc_field_l,
    $REAL* inc_field_u) {
        
        //  This method updates the IS fields.
        //
        //  Args:
        //      nwx, nwy, nwz: Number of cells in respective directions.
        //      n: Number of boundary cells.
        //      offset: Helps distinguishing the H nodes from the E nodes.
        //      nwl, nwm: Number of cells in the 
        //      face: Determines which face of the cube to calculate for.
        //      co: Coefficient used by gprMax update equations which
        //      is specific to the field component being updated.
        //      sign_l, sign_u: Lower and upper signs for the precursor fields
        //      lookup_id: The index value of the component of the main_grid fields.
        //      pre_coeff: The coefficient of the precursor fields.
        //      updatecoeffs: The update coefficients of the field.
        //      ID: The ID of the components
        //      field: the sub_grid fields to be updated
        //      inc_field_l, inc_field_u: Lower and upper precursor fields.



        // Current Thread Index
        int idx = blockIdx.x * blockDim.x + threadIdx.x;

        int l, m, i1, j1, k1, i2, j2, k2, field_material_l, field_material_u, inc_i, inc_j;
        double inc_l, inc_u, f_l, f_u;
        int n_o;

        // For inner Faces H nodes are 1 cell before n_boundary_cell
        n_o = n + offset;

        // Linear Index to subscript 
        l = idx / ($NZ_FIELDS * $NY_FIELDS); 
        m = idx % ($NZ_FIELDS * $NY_FIELDS);

        if(l >= n && l < (nwl + n) && m >= n && m < (nwm + n)) {
            if(face == 1) {
                i1 = l; j1 = m; k1 = n_o;
                i2 = l; j2 = m; k2 = n + nwz;
            }
            else if(face == 2) {
                i1 = n_o; j1 = l; k1 = m;
                i2 = n + nwx; j2 = l; k2 = m;
            }
            else {
                i1 = l; j1 = n_o; k1 = m;
                i2 = l; j2 = n + nwy; k2 = m;
            }
            
            inc_i = l - n;
            inc_j = m - n;

            // Precursor Field index
            int pre_index = (inc_i * pre_coeff) + inc_j;

            field_material_l = ID[INDEX4D_ID(lookup_id, i1, j1, k1)];
            inc_l = inc_field_l[pre_index];

            // Additional Field at i, j, k
            f_l = updatecoeffs[INDEX2D_MAT(field_material_l, co)] * inc_l * sign_l;
            
            // Setting the new value
            field[INDEX3D_FIELDS(i1, j1, k1)] += f_l;

            field_material_u = ID[INDEX4D_ID(lookup_id, i2, j2, k2)];
            inc_u = inc_field_u[pre_index];

            f_u = updatecoeffs[INDEX2D_MAT(field_material_u, co)] * inc_u * sign_u;
            field[INDEX3D_FIELDS(i2, j2, k2)] += f_u;
        }
    }
""")
