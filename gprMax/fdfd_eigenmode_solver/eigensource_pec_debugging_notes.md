# FDFD Eigenmode Source PEC Debugging Notes

This note records the debugging path for the PEC-loaded eigensource failures and the fixes that came out of it. The short version is:

- The FDFD mode solve had to understand PEC as Yee-component constraints, not as ordinary high-epsilon material.
- The eigensource had to pass the solver correct PEC constraints from the final gprMax Yee grid.
- Modal field injection direction was not the root problem. The original propagation signs were right.
- The remaining impurity in all `+` PEC-loaded cases was caused by asymmetric PEC box edge assignment in the FDTD geometry build, not by the FDFD mode itself.

The final fixes touch:

- `gprMax/fdfd_eigenmode_solver/fdfd_2d_mode_solver.py`
- `gprMax/sources.py`
- `gprMax/cython/eigenmode_source.pyx`
- `gprMax/user_objects/cmds_geometry/box.py`
- `gprMax/cython/geometry_primitives.pyx`

## Initial Failure

The starting test was a y-propagating dielectric guide with a PEC strip next to the guide:

```text
#box: 0.007 0.000 0.007 0.011 0.064 0.010 guide_core
#box: 0.003 0.000 0.010 0.015 0.064 0.011 pec
#eigenmode_source: 0.002 0.025 0.0025 0.016 0.025 0.0155 + 2 40e9 eig_pulse
```

That case solved a reasonable mode. Moving the PEC to the opposite side produced a wrong solved eigenmode:

```text
#box: 0.007 0.000 0.007 0.011 0.064 0.010 guide_core
#box: 0.003 0.000 0.006 0.015 0.064 0.007 pec
#eigenmode_source: 0.002 0.025 0.0025 0.016 0.025 0.0155 + 2 40e9 eig_pulse
```

Physically those two cross-sections should be mirrors. The effective index and modal field should be nearly the same, just flipped in the transverse direction.

## Test Matrix That Exposed the Real Pattern

The original y-only cases were not enough to separate solver, source, direction, and PEC-side bugs. A wider test matrix was created:

- `dielectric_ridge`: six tests, `x/y/z` and `+/-`, with two dielectric rectangles.
- `pec_loaded`: six tests, `x/y/z` and `+/-`, with a dielectric block and a PEC block.

The important result was:

- All ridge cases worked.
- All PEC-loaded `x-`, `y-`, `z-` cases worked.
- All PEC-loaded `x+`, `y+`, `z+` cases propagated in the correct direction but showed modal impurity.

That result eliminated several possibilities:

- Not a general sign error: direction was correct in all cases.
- Not a y-axis-only transform bug: x, y, and z all showed the same plus/minus split.
- Not a general FDFD mode-selection bug: the ridge cases and the PEC `-` cases solved and injected cleanly.
- Not a bad solved PEC mode for the final issue: the `+/-` PEC modes had matching `neff` and mirrored field plots.

The bug had to be in the coupling between PEC-constrained modal fields and the final FDTD Yee grid.

## FDFD Solver Fixes

### Use the Same EM Constants as gprMax

The FDFD solver originally used rounded local constants:

```python
self.epsilon0 = 8.85e-12
self.mu0 = 1.26e-6
self.c = 1 / np.sqrt(self.epsilon0 * self.mu0)
self.eta0 = np.sqrt(self.mu0 / self.epsilon0)
```

That is not catastrophic by itself, but it is unnecessary inconsistency. The solver now uses the same constants as the running gprMax simulation:

```python
self.epsilon0 = config.sim_config.em_consts["e0"]
self.mu0 = config.sim_config.em_consts["m0"]
self.c = config.sim_config.em_consts["c"]
self.eta0 = config.sim_config.em_consts["z0"]
```

This removes small normalization and phase inconsistencies between modal solve and FDTD injection.

### Treat PEC as Constrained Electric DOFs

PEC should not be represented as a material with huge or infinite permittivity inside the eigenproblem. That creates artificial localized PEC material modes and can dominate the eigensolver.

The FDFD solver instead builds masks:

- `pec_ex_mask`
- `pec_ey_mask`
- `pec_ez_mask`

Those masks identify electric degrees of freedom that must be removed from the solve or zeroed after field reconstruction. Matrix assembly uses harmless finite placeholders at masked locations, while the masks carry the actual PEC physics.

### Combine Explicit Masks With Non-Finite Material Samples

One early mistake was that providing an explicit PEC mask replaced the implicit non-finite material mask. That made the behavior fragile: a caller could accidentally lose constraints already encoded in the component-sampled material arrays.

The mask logic now ORs both sources:

```python
mask = np.zeros(values.shape, dtype=bool)
if default:
    mask |= ~np.isfinite(values)
if explicit_mask is not None:
    explicit_mask = np.asarray(explicit_mask, dtype=bool)
    if explicit_mask.shape != values.shape:
        raise ValueError(...)
    mask |= explicit_mask
return mask
```

This matters because the gprMax integration uses both:

- component-sampled `G.ID` material values at Yee E locations;
- supplemental cell-centered PEC masks to repair one-sided PEC box sampling.

### Reject PEC-Localized Spurious Modes

The solver also solves extra candidate modes and rejects modes with too much energy near PEC constraints:

```python
def solve(
        self,
        reject_spurious=True,
        extra_modes=8,
        max_pec_neighbor_energy_fraction=0.35,
):
```

This is important for PEC-loaded structures because artificial high-gradient modes near the conductor can be numerically attractive to the sparse eigensolver. The accepted mode set is chosen from candidates whose transverse electric energy is not dominated by PEC-neighbor cells.

## Passing the Right PEC Geometry Into FDFD

The source extracts material tensors only after the gprMax grid is built:

```python
def grid_init(self, G):
    ...
    self._extract_complex_property_tensors(G, electric=True)
    self._extract_complex_property_tensors(G, electric=False)
    self._solve_eigenmode(G)
```

That timing is critical. Before `grid_init()`, the final Yee `ID` arrays do not exist, so the source cannot know the actual material sampled by each E or H component.

For FDFD, `EigenmodeSource._solve_eigenmode()` now passes explicit component masks:

```python
pec_ex_mask, pec_ey_mask, pec_ez_mask = self._cell_pec_component_masks(G)

solver = FDFD_2D_mode_solver(
    ...
    pec_ex_mask=pec_ex_mask,
    pec_ey_mask=pec_ey_mask,
    pec_ez_mask=pec_ez_mask,
)
```

The helper `_slice_cell_pec_mask()` extracts PEC cells from `G.solid` around the source plane. It deliberately looks at both adjacent normal cells:

```python
normal_indices = [
    index
    for index in (self.plane_index - 1, self.plane_index)
    if 0 <= index < G.solid.shape[self.normal_axis]
]
```

Then `_cell_pec_component_masks()` expands those cell-centered PEC cells onto local FDFD component masks. This was needed because non-averaged PEC boxes in gprMax were one-sided at Yee faces. Without this supplement, the modal solve saw one PEC side differently from the mirrored side.

## Direction and Modal Handedness

The original modal injection signs were tested heavily. A tempting fix was to flip the y-plus sign, because it made one visual case look different. That was wrong.

The decisive tests were dielectric-only guides in all directions:

- `x+` and `x-` injected in the correct directions.
- `y+` and `y-` injected in the correct directions.
- `z+` and `z-` injected in the correct directions.

Changing a `+` sign fixed nothing fundamental and could inject the mode in the wrong direction. The correct approach was to keep the propagation signs and fix the PEC/Yee geometry mismatch.

The code still handles local basis handedness:

```python
if self._modal_basis_handedness() < 0:
    self.modal_h = [-field for field in self.modal_h]
```

That is separate from propagation direction. It corrects the local transverse basis orientation so the modal Poynting direction remains consistent when mapping local FDFD fields back to global `Ex/Ey/Ez/Hx/Hy/Hz`.

## Modal Electric Source Update: False Lead and Cleanup

One intermediate fix tried to pass the FDFD PEC masks into the Cython electric source update:

```python
update_eigenmode_electric(..., pec_Ex, pec_Ey, pec_Ez, ...)
```

The idea was simple: do not inject electric TF/SF corrections at modal E samples that the FDFD solver considered PEC-constrained.

That was not the right long-term fix.

Reason:

- The FDFD masks describe local modal-solver constraints.
- The FDTD update should be governed by the actual FDTD `G.ID` and `G.updatecoeffsE` at the target Yee node.
- Reusing the FDFD mask inside the time-domain source update can suppress active FDTD nodes that are not actually PEC in the current Yee grid.

This was visible in diagnostics:

- Some nodes under the FDFD mask still had non-zero FDTD electric update coefficients.
- Suppressing those nodes slightly changed the field but did not fix the impurity.

The mask arguments were removed from `update_eigenmode_electric()`. The update now applies the standard TF/SF correction using `G.ID` and `G.updatecoeffsE`. PEC nodes naturally receive zero update coefficients when the grid geometry is correct.

## Final Root Cause: PEC Box Edge Assignment

The final plus/minus pattern came from `build_box()` for non-averaged boxes.

For rigid/non-averaged boxes, the previous implementation mainly assigned component IDs at the lower corner of each cell and patched a limited set of high-side faces after the loop. That made PEC boxes asymmetric at Yee edges:

- A PEC above the dielectric constrained the guide/PEC interface.
- A PEC below the dielectric left the guide/PEC interface active and constrained the outer side instead.

The generated PEC-loaded test cases coupled direction and PEC side:

- `+` cases had PEC on the lower transverse side.
- `-` cases had PEC on the upper transverse side.

So all `+` cases failed and all `-` cases worked. The root was not the sign of `+`; it was the fact that all `+` tests put PEC on the side whose tangential E interface was not actually constrained in the FDTD grid.

The fix is PEC-specific. In `Box.build()`:

```python
constrain_all_edges = any(getattr(material, "se", 0) == float("inf") for material in materials)
```

That flag is passed to `build_box()`. In `geometry_primitives.pyx`, non-averaged PEC boxes now call `build_voxel()` for every voxel:

```python
elif constrain_all_edges:
    for i in range(xs, xf):
        for j in range(ys, yf):
            for k in range(zs, zf):
                build_voxel(i, j, k, numID, numIDx, numIDy, numIDz,
                            averaging, solid, rigidE, rigidH, ID)
```

`build_voxel()` assigns all relevant Yee electric and magnetic component IDs around the voxel, including high-side edges. That makes PEC boundary constraints symmetric for top/bottom, left/right, and front/back PEC placement.

This was intentionally limited to infinite-conductivity materials. Ordinary dielectric boxes keep the existing behavior.

## Validation

### Geometry-Only Constraint Check

After the PEC box edge fix, a geometry-only inspection checked all six PEC-loaded cases:

- `pec_x_plus`
- `pec_x_minus`
- `pec_y_plus`
- `pec_y_minus`
- `pec_z_plus`
- `pec_z_minus`

For each source, the diagnostic mapped local solver masks back to global components and checked whether any FDFD-PEC-constrained tangential E nodes still had non-zero FDTD electric update coefficients.

Result:

```text
pec_x_plus:  Ex/Ey/Ez tangential active_under_solver_mask count = 0
pec_x_minus: Ex/Ey/Ez tangential active_under_solver_mask count = 0
pec_y_plus:  Ex/Ey/Ez tangential active_under_solver_mask count = 0
pec_y_minus: Ex/Ey/Ez tangential active_under_solver_mask count = 0
pec_z_plus:  Ex/Ey/Ez tangential active_under_solver_mask count = 0
pec_z_minus: Ex/Ey/Ez tangential active_under_solver_mask count = 0
```

That was the key structural validation. The FDFD solver and the FDTD Yee grid now agree on which tangential E nodes are PEC-constrained.

### Short Propagation Rerun

A shortened `pec_x_plus` run was used to avoid overwriting completed full outputs. The transverse profile correlation was compared against the original `pec_x_plus` and clean `pec_x_minus` snapshots.

The metric used was simple:

- Compute `|E|` in the `xz` snapshot at `120 ps`.
- For columns above 25 percent of peak column energy, normalize each transverse profile.
- Compare each profile with a central reference profile.
- A purer mode keeps a more consistent transverse shape as it propagates.

Result:

```text
original pec_x_plus:  min/mean correlation = 0.716845 / 0.971843
patched  pec_x_plus:  min/mean correlation = 0.876512 / 0.989702
original pec_x_minus: min/mean correlation = 0.839670 / 0.990696
```

The patched `x+` case reached the same mean profile consistency as the clean `x-` case and improved the worst-profile correlation beyond the original `x-` value.

### Rebuild Command

After Cython changes:

```powershell
& C:\Users\Traveler\anaconda3\Scripts\conda.exe run -n gprMax python setup.py build
```

Both relevant rebuilds completed successfully:

- `gprMax.cython.eigenmode_source`
- `gprMax.cython.geometry_primitives`

## Practical Lessons

### Do Not Fix Direction Bugs With Visual Sign Flips

A sign flip may make one plot look more plausible while breaking propagation direction. The correct isolation test is:

- dielectric-only guide;
- all three axes;
- both `+` and `-`;
- compare propagation direction and modal purity separately.

Only after those pass should PEC-specific behavior be debugged.

### Keep Solver Constraints and FDTD Constraints Conceptually Separate

The FDFD solver needs explicit constraints to remove PEC electric DOFs. The FDTD source update should not blindly reuse those masks. Once the FDTD geometry is correct, PEC nodes already have the right material IDs and zero electric update coefficients.

Correct layering:

- FDFD: use PEC masks to constrain modal DOFs.
- FDTD geometry: assign PEC to all Yee E edges that physically lie on/in a PEC object.
- Modal injection: apply TF/SF corrections at target FDTD nodes and let `G.updatecoeffsE/H` handle material response.

### Component-Sampled Yee Geometry Is the Source of Truth

For eigensource integration, cell-centered geometry is not enough. The source plane cuts through Yee fields, not cell centers. Any conductor or dielectric interface must be checked at the actual component locations used by:

- `G.ID[0]` for `Ex`
- `G.ID[1]` for `Ey`
- `G.ID[2]` for `Ez`
- `G.ID[3]` for `Hx`
- `G.ID[4]` for `Hy`
- `G.ID[5]` for `Hz`

The most useful diagnostic was not just plotting the field. It was asking:

```text
Does the FDFD solver think this tangential E node is PEC?
If yes, does the FDTD update coefficient at the corresponding Yee node equal zero?
```

Before the final fix, the answer was sometimes no. After the PEC edge fix, the answer is consistently yes.

## Current Expected Behavior

With the fixes in place:

- Dielectric ridge `x/y/z +/-` should still work.
- PEC-loaded `x/y/z -` should still work.
- PEC-loaded `x/y/z +` should no longer show the old bottom-PEC impurity.
- `+` propagation signs should remain unchanged.
- Mirrored top/bottom PEC placements should produce mirrored modes with matching effective index.

If a future PEC-loaded case fails, first check whether the FDFD and FDTD PEC constraints agree at the source plane before changing modal signs.
