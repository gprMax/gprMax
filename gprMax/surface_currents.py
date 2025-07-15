import numpy as np
import matplotlib.pyplot as plt

def get_surface_currents(filename, padding = 60):
    # Load file
    data = np.load(filename, allow_pickle=True)

    electric = data['electric']
    magnetic = data['magnetic']
    
    face_elec = electric[0]
    face_mag = magnetic[0]
    t = len(face_elec)
    Js_0 = np.zeros((t, 2, padding, padding))  # Js[t, 0] = Jy, Js[t, 1] = Jz
    Ms_0 = np.zeros((t, 2, padding, padding))  # Ms[t, 0] = My, Ms[t, 1] = Mz
    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag)):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap
        Hz = Hx[0, :, :]
        Hy = Hy[0, :, :]
        Ez = Ex[0, :, :]
        Ey = Ey[0, :, :]

        Ms_0[t, 0] = Ez        # My
        Ms_0[t, 1] = -Ey       # Mz

        Js_0[t, 0] = -Hz       # Jy
        Js_0[t, 1] = Hy        # Jz
    
    face_elec = electric[1]
    face_mag = magnetic[1]
    t = len(face_elec)
    Js_1 = np.zeros((t, 2, padding, padding))  # Js[t, 0] = Jx, Js[t, 1] = Jy
    Ms_1 = np.zeros((t, 2, padding, padding))  # Ms[t, 0] = Mx, Ms[t, 1] = My
    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag)):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap

        Hx = Hx[:, :, 0]
        Hy = Hy[:, :, 0]
        Ex = Ex[:, :, 0]
        Ey = Ey[:, :, 0]

        Ms_1[t, 0] = -Ey        # Mx
        Ms_1[t, 1] = Ex       # My

        Js_1[t, 0] = Hy       # Jx
        Js_1[t, 1] = -Hx        # Jy    

        
    face_elec = electric[2]
    face_mag = magnetic[2]
    t = len(face_elec)
    Js_2 = np.zeros((t, 2, padding, padding))   # Js[t, 0] = Jx, Js[t, 1] = Jz
    Ms_2 = np.zeros((t, 2, padding, padding))  # Ms[t, 0] = Mx, Ms[t, 1] = Mz
    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag)):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap

        Hx = Hx[:, 0, :]
        Hz = Hy[:, 0, :]
        Ex = Ex[:, 0, :]
        Ez = Ey[:, 0, :]

        Ms_2[t, 0] = Ez        # Mx
        Ms_2[t, 1] = -Ex       # Mz

        Js_2[t, 0] = -Hz       # Jx
        Js_2[t, 1] = Hx        # Jz    

    face_elec = electric[3]
    face_mag = magnetic[3]
    t = len(face_elec)
    Js_3 = np.zeros((t, 2, padding, padding))  # Js[t, 0] = Jx, Js[t, 1] = Jy
    Ms_3 = np.zeros((t, 2, padding, padding))  # Ms[t, 0] = Mx, Ms[t, 1] = My
    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag)):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap

        Hx = Hx[:, :, 0]
        Hy = Hy[:, :, 0]
        Ex = Ex[:, :, 0]
        Ey = Ey[:, :, 0]

        Ms_3[t, 0] = Ey        # Mx
        Ms_3[t, 1] = -Ex       # My

        Js_3[t, 0] = -Hy       # Jx
        Js_3[t, 1] = Hx        # Jy    

    face_elec = electric[4]
    face_mag = magnetic[4]
    t = len(face_elec)
    Js_4 = np.zeros((t, 2, padding, padding))   # Js[t, 0] = Jx, Js[t, 1] = Jz
    Ms_4 = np.zeros((t, 2, padding, padding))  # Ms[t, 0] = Mx, Ms[t, 1] = Mz
    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag)):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap

        Hx = Hx[:, 0, :]
        Hz = Hy[:, 0, :]
        Ex = Ex[:, 0, :]
        Ez = Ey[:, 0, :]

        Ms_4[t, 0] = -Ez        # Mx
        Ms_4[t, 1] = Ex       # Mz

        Js_4[t, 0] = Hz       # Jx
        Js_4[t, 1] = -Hx        # Jz 

    face_elec = electric[5]
    face_mag = magnetic[5]
    t = len(face_elec)
    Js_5 = np.zeros((t, 2, padding, padding))  # Js[t, 0] = Jy, Js[t, 1] = Jz
    Ms_5 = np.zeros((t, 2, padding, padding))  # Ms[t, 0] = My, Ms[t, 1] = Mz
    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag)):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap
        Hz = Hx[0, :, :]
        Hy = Hy[0, :, :]
        Ez = Ex[0, :, :]
        Ey = Ey[0, :, :]

        Ms_5[t, 0] = -Ez        # My
        Ms_5[t, 1] = Ey       # Mz

        Js_5[t, 0] = Hz       # Jy
        Js_5[t, 1] = -Hy        # Jz

    Js_faces = [Js_0, Js_1, Js_2, Js_3, Js_4, Js_5]
    Ms_faces = [Ms_0, Ms_1, Ms_2, Ms_3, Ms_4, Ms_5]

    return Js_faces, Ms_faces


def generate_surface_coordinates(N=60, cube_size=1.0):
    """
    Returns a NumPy array of shape (6, 3, N, N) where:
    - 6 is the number of faces
    - 3 corresponds to X, Y, Z coordinates
    - N x N is the spatial resolution of each face
    """
    L = cube_size
    lin = np.linspace(-L/2, L/2, N)
    coords = np.zeros((6, 3, N, N), dtype=np.float32)

    # Face 0: -x (Y-Z plane at x = -L/2)
    Y, Z = np.meshgrid(lin, lin, indexing='ij')
    coords[0, 0] = -L/2
    coords[0, 1] = Y
    coords[0, 2] = Z

    # Face 1: +x (Y-Z plane at x = +L/2)
    coords[1, 0] = +L/2
    coords[1, 1] = Y
    coords[1, 2] = Z

    # Face 2: -y (X-Z plane at y = -L/2)
    X, Z = np.meshgrid(lin, lin, indexing='ij')
    coords[2, 0] = X
    coords[2, 1] = -L/2
    coords[2, 2] = Z

    # Face 3: +y (X-Z plane at y = +L/2)
    coords[3, 0] = X
    coords[3, 1] = +L/2
    coords[3, 2] = Z

    # Face 4: -z (X-Y plane at z = -L/2)
    X, Y = np.meshgrid(lin, lin, indexing='ij')
    coords[4, 0] = X
    coords[4, 1] = Y
    coords[4, 2] = -L/2

    # Face 5: +z (X-Y plane at z = +L/2)
    coords[5, 0] = X
    coords[5, 1] = Y
    coords[5, 2] = +L/2

    return coords


def compute_far_field_at_point(Js_faces, Ms_faces, surface_coords, r_obs, theta, phi, wavelength, dx, dy):
    """
    Computes the far-field electric field vector at a single observation point (r, θ, φ).

    Parameters:
        Js_faces: list or array of shape (6, T, 2, N, N) for electric surface currents
        Ms_faces: same as above for magnetic surface currents
        surface_coords: np.array of shape (6, 3, N, N) giving X, Y, Z coordinates for each face
        r_obs: radial distance to observation point
        theta, phi: spherical coordinates of observation point
        wavelength: operating wavelength
        dx, dy: spatial resolution on the face

    Returns:
        E_far: complex 3D vector (numpy array of shape (3,)) representing the electric field
    """
    k = 2 * np.pi / wavelength
    r_hat = np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    E_far = np.zeros(3, dtype=complex)

    for face in range(6):
        Js_face = Js_faces[face]  # shape: (T, 2, N, N)
        Ms_face = Ms_faces[face]
        X = surface_coords[face, 0]  # shape: (N, N)
        Y = surface_coords[face, 1]
        Z = surface_coords[face, 2]

        N = X.shape[0]

        for i in range(N):
            for j in range(N):
                r_prime = np.array([X[i, j], Y[i, j], Z[i, j]])
                R = r_obs - np.dot(r_hat, r_prime)
                phase = np.exp(1j * k * R)

                Js_vec = np.array([Js_face[-1, 0, i, j], Js_face[-1, 1, i, j], 0.0])
                Ms_vec = np.array([Ms_face[-1, 0, i, j], Ms_face[-1, 1, i, j], 0.0])

                term = np.cross(r_hat, Js_vec + np.cross(r_hat, Ms_vec)) * phase
                E_far += term * dx * dy

    prefactor = 1j * k / (4 * np.pi * r_obs) * np.exp(-1j * k * r_obs)
    E_far *= prefactor
    return E_far


if __name__ == "__main__":
    # Js_0, Js_1, Js_2, Js_3, Js_4, Js_5  = get_surface_currents('nfft_snapshots_model1.npz')
    # print(Js_0.shape, Js_5.shape)
    
    coords = generate_surface_coordinates(60)
    print(coords[1][2][1][4])


    ############# Generate coordinates
    surface_coords = generate_surface_coordinates(N=60, cube_size=1.0)
    dx = dy = 1.0 / 60

    # Compute far field at (r=10, θ=60°, φ=30°)
    E_far = compute_far_field_at_point(
        Js_faces, Ms_faces, surface_coords,
        r_obs=10.0,
        theta=np.deg2rad(60),
        phi=np.deg2rad(30),
        wavelength=1.0,
        dx=dx, dy=dy
    )

    print("Far-field E vector:", E_far)
