import numpy as np

# Load the saved .npz file
data = np.load('nfft_snapshots_model1.npz', allow_pickle=True)

electric = data['electric']
magnetic = data['magnetic']

print(f"Loaded data with {len(electric)} electric faces and {len(magnetic)} magnetic faces.\n")

# for i, (face_elec, face_mag) in enumerate(zip(electric, magnetic), 1):
#     print(f"Face {i}:")
#     print(f"  Number of timesteps: {len(face_elec)}")

#     for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag), 1):
#         Ex, Ey, Ez = e_snap
#         Hx, Hy, Hz = m_snap

#         print(f"    Timestep {t}:")
#         print(f"      Ex shape: {Ex.shape}, Ey shape: {Ey.shape}, Ez shape: {Ez.shape}")
#         print(f"      Hx shape: {Hx.shape}, Hy shape: {Hy.shape}, Hz shape: {Hz.shape}")

#     print()

face_elec = electric[5]
e_snap = face_elec[200]
Ex, Ey, Ez = e_snap
print(f"      Ex shape: {Ex.shape}, Ey shape: {Ey.shape}, Ez shape: {Ez.shape}")


    #   Ex shape: (1, 60, 60), Ey shape: (1, 60, 60), Ez shape: (1, 60, 60)
    #   Ex shape: (60, 60, 1), Ey shape: (60, 60, 1), Ez shape: (60, 60, 1)
    #   Ex shape: (60, 1, 60), Ey shape: (60, 1, 60), Ez shape: (60, 1, 60)
    #   Ex shape: (60, 60, 1), Ey shape: (60, 60, 1), Ez shape: (60, 60, 1)
    #   Ex shape: (60, 1, 60), Ey shape: (60, 1, 60), Ez shape: (60, 1, 60)
    #   Ex shape: (1, 60, 60), Ey shape: (1, 60, 60), Ez shape: (1, 60, 60)