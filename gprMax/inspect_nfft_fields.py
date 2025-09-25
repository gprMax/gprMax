import numpy as np
import matplotlib.pyplot as plt
# Load file
data = np.load('nfft_snapshots_model1.npz', allow_pickle=True)

electric = data['electric']
magnetic = data['magnetic']

print(f"Loaded data with {len(electric)} electric faces and {len(magnetic)} magnetic faces.\n")

for i, (face_elec, face_mag) in enumerate(zip(electric, magnetic), 1):
    print(f"Face {i}:")
    print(f"  Timesteps: {len(face_elec)}")

    for t, (e_snap, m_snap) in enumerate(zip(face_elec, face_mag), 1):
        Ex, Ey, Ez = e_snap
        Hx, Hy, Hz = m_snap

        print(f"    Timestep {t}:")

        def inspect_field(name, arr):
            if arr.size == 0:
                print(f"      {name}: EMPTY array!")
            elif np.all(arr == 0):
                print(f"      {name}: all zeros, shape={arr.shape}")
            else:
                print(f"      {name}: shape={arr.shape}, min={arr.min():.4e}, max={arr.max():.4e}")

        # Electric fields
        inspect_field("Ex", Ex)
        inspect_field("Ey", Ey)
        inspect_field("Ez", Ez)

        # Magnetic fields
        inspect_field("Hx", Hx)
        inspect_field("Hy", Hy)
        inspect_field("Hz", Hz)

    print()
