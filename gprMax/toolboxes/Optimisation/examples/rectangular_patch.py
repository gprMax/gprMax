"""Optimise a magnetic-frill-fed rectangular patch: python rectangular_patch.py

Edit sections 1-4, just as in start_here.py. Supporting checks are at the end.
This is an adaptation of Jin & Rahmat-Samii (2005), Section III:
https://doi.org/10.1109/TAP.2005.858842
The frequency, board and substrate follow the paper. The search bounds,
starting design and equivalent coaxial feed below are our choices. The two
Table I geometries are included as references. See rectangular_patch.md.
"""

import math
from pathlib import Path

import numpy as np

import gprMax
from gprMax.toolboxes.Optimisation import Integer, LocalExecutor, optimise, simulate

# 1. CHOOSE WHAT MAY CHANGE AND WHAT YOU WANT TO ACHIEVE.
# These are physical dimensions in MILLIMETRES, not numbers of mesh cells.
# Keeping the same ranges when refining the mesh preserves the design choices.
# L >= 26 mm keeps every offset up to 12 mm at least 1 mm inside the patch.
# Even L/W place both the patch centre and feed centreline exactly on this
# model's 1 mm design lattice. step=2 is shared by every optimiser adapter.
PARAMETERS = {
    "length_mm": Integer(26, 36, "mm", step=2),
    "width_mm": Integer(12, 44, "mm", step=2),
    "feed_offset_mm": Integer(1, 12, "mm"),
}
STARTING_DESIGN = {"length_mm": 30, "width_mm": 30, "feed_offset_mm": 7}
# Figure 2 measures x from the patch CENTRE. Positive offsets go toward -x.
# The starting and published designs retain their previous physical positions;
# an old edge inset converts as: centre offset = length/2 - edge inset.
PUBLISHED_DESIGNS = {
    "antenna_I": {"length_mm": 30, "width_mm": 18, "feed_offset_mm": 4},
    "antenna_II": {"length_mm": 30, "width_mm": 44, "feed_offset_mm": 12},
}
TARGET_FREQUENCY_HZ = 3.1e9
ACCEPTABLE_S11_DB = -30.0  # Stop after a complete population meets this score.
CELL_SIZE_M = 0.001
TIME_WINDOW_S = 100e-9  # Approximately 10 MHz independent FFT bins.
BOARD_SIZE_M = 0.060
SUBSTRATE_HEIGHT_M = 0.003
SUBSTRATE_PERMITTIVITY = 2.2
AIR_PADDING_M = 0.020  # Distance from board edges to domain boundary.
PML_THICKNESS_M = 0.010  # Included in AIR_PADDING_M.
PROBE_RADIUS_M = 0.0002  # Assumed; not specified in the paper.
COAX_IMPEDANCE_OHM = 50.0
COAX_FILLER_PERMITTIVITY = 1.0  # Assumed air-filled coax; NOT the substrate.
RESULTS = Path(__file__).with_name("patch_results")  # Use a new folder per run.


# 2. BUILD ONE gprMax MODEL FROM THE TRIAL VALUES.
def build_model(parameters):
    """Convert the proposed millimetres into a complete, unexecuted Scene.

    The board and substrate remain fixed. Only the patch and probe position
    change. All gprMax coordinates below are metres.
    """
    validate_model_settings(parameters)  # Supporting checks are at the file end.
    length_m = parameters["length_mm"] * 1e-3
    width_m = parameters["width_mm"] * 1e-3
    # The requested offset is from the actual patch centre. For odd lengths,
    # the supporting helper rounds the feed onto the fixed 1 mm design lattice.
    # Both requested and realised offsets are recorded by evaluate().
    offset_m = realised_feed_offset_mm(parameters) * 1e-3

    domain_xy = BOARD_SIZE_M + 2 * AIR_PADDING_M
    centre = domain_xy / 2
    ground_z = AIR_PADDING_M
    patch_z = ground_z + SUBSTRATE_HEIGHT_M

    # Place the patch on the 1 mm design lattice. An odd dimension moves its
    # centre by -0.5 mm. This convention stays identical on a finer solver grid;
    # it avoids silently rounding two edges into an unintended patch length.
    patch_x0 = centre - math.ceil(parameters["length_mm"] / 2) * 1e-3
    patch_y0 = centre - math.ceil(parameters["width_mm"] / 2) * 1e-3
    patch_centre_x = patch_x0 + length_m / 2
    feed_x = patch_centre_x - offset_m
    feed_y = patch_y0 + math.floor(parameters["width_mm"] / 2) * 1e-3

    scene = gprMax.Scene()
    scene.add(gprMax.Domain(p1=(domain_xy, domain_xy, 2 * AIR_PADDING_M + SUBSTRATE_HEIGHT_M)))
    scene.add(gprMax.Discretisation(p1=(CELL_SIZE_M,) * 3))
    scene.add(gprMax.TimeWindow(time=TIME_WINDOW_S))
    scene.add(gprMax.PMLThickness(thickness=round(PML_THICKNESS_M / CELL_SIZE_M)))
    scene.add(gprMax.Material(er=SUBSTRATE_PERMITTIVITY, se=0, mr=1, sm=0, id="substrate"))

    # Build dielectric BEFORE metal. Zero-thickness PEC plates form the ground
    # and patch; dielectric loss and metal loss are deliberately zero here.
    scene.add(
        gprMax.Box(
            p1=(AIR_PADDING_M, AIR_PADDING_M, ground_z),
            p2=(AIR_PADDING_M + BOARD_SIZE_M, AIR_PADDING_M + BOARD_SIZE_M, patch_z),
            material_id="substrate",
        )
    )
    scene.add(
        gprMax.Plate(
            p1=(AIR_PADDING_M, AIR_PADDING_M, ground_z),
            p2=(AIR_PADDING_M + BOARD_SIZE_M, AIR_PADDING_M + BOARD_SIZE_M, ground_z),
            material_id="pec",
        )
    )
    scene.add(
        gprMax.Plate(
            p1=(patch_x0, patch_y0, patch_z),
            p2=(patch_x0 + length_m, patch_y0 + width_m, patch_z),
            material_id="pec",
        )
    )

    # The vertical inner conductor has the same 3 mm length as the substrate
    # thickness: ground surface to patch underside. It has NO voltage-source gap.
    # The frill analytically represents the small insulating coaxial aperture
    # around its bottom endpoint. The grid's PEC ground plane remains uncut;
    # this equivalent source supplies the aperture voltage and coaxial loading.
    # A bare wire without that frill source would be a shorting pin.
    scene.add(
        gprMax.ThinWire(
            p1=(feed_x, feed_y, ground_z),
            p2=(feed_x, feed_y, patch_z),
            radius=PROBE_RADIUS_M,
        )
    )
    scene.add(gprMax.Waveform(wave_type="ricker", amp=1, freq=TARGET_FREQUENCY_HZ, id="pulse"))
    scene.add(
        gprMax.MagneticFrillSource(
            p1=(feed_x, feed_y, ground_z),
            polarisation="z",
            zcoax=COAX_IMPEDANCE_OHM,
            waveform_id="pulse",
        )
    )
    return scene  # The toolbox executes it and collects the automatic port output.


# 3. SAY HOW TO SCORE THE COMPLETED SIMULATION: SMALLER IS BETTER.
def evaluate(parameters, output):
    """Minimise S11 in dB at 3.1 GHz; -20 dB is better than -10 dB.

    The criterion is matching at one fixed frequency. Locating the frequency
    of the deepest dip, or maximising bandwidth, would be different objectives.
    Poorly matched antennas remain valid candidates and receive a poor score.
    """
    # gprMax numbers magnetic frills in source order: the first is frill1.
    # The qualified name explicitly selects its solver-written terminal output.
    spectrum = output.port("frills/frill1")
    check_spectrum(spectrum)  # Supporting data-quality checks at the end.
    s11_at_target = spectrum.at(TARGET_FREQUENCY_HZ)
    # THIS is the optimisation criterion. The floor only makes log(0) finite.
    score_db = 20 * np.log10(max(abs(s11_at_target), 1e-12))
    output.save_npz(
        "s11.npz",
        frequency_hz=spectrum.frequency,
        s11=spectrum.s11,
        valid=spectrum.valid,
        target_hz=TARGET_FREQUENCY_HZ,
        s11_at_target=s11_at_target,
        score_db=score_db,
        requested_feed_offset_mm=parameters["feed_offset_mm"],
        realised_feed_offset_mm=realised_feed_offset_mm(parameters),
    )
    return float(score_db)


# 4. CHECK ONE DESIGN, THEN RUN THE SEARCH.
def run_example(
    directory=RESULTS,
    *,
    optimiser="pso",
    evaluations=40,
    seed=7,
    population_size=10,
    execution=None,
    target_value=ACCEPTABLE_S11_DB,
):
    """Use the public interface for both the starting design and the search.

    Change the optimiser to "ga", "de", "rf" or "tpe" without rewriting the
    model or criterion. evaluations counts candidate requests; repeated designs
    can occur. This is a budget, not convergence. PSO uses whole populations.
    Pass a LocalPool as execution to evaluate a population on several workers.
    target_value is an acceptable S11 in dB; None uses the evaluation limit.
    This starting simulation checks the model; it does not seed the optimiser.
    """
    directory = Path(directory)
    execution = execution or LocalExecutor(cpu_threads=2, timeout=1800)
    starting_output = simulate(
        model=build_model,
        parameters=STARTING_DESIGN,
        directory=directory / "starting_design",
        execution=execution,
    )
    # This independent check does not seed a particle into the PSO population.
    starting_score = evaluate(STARTING_DESIGN, starting_output)
    print(f"Starting design: S11 at 3.1 GHz = {starting_score:.2f} dB", flush=True)
    result = optimise(
        parameters=PARAMETERS,
        model=build_model,
        objective=evaluate,
        directory=directory / "search",
        optimiser=optimiser,
        evaluations=evaluations,
        seed=seed,
        population_size=population_size,
        execution=execution,
        target_value=target_value,
    )
    print(f"Best evaluated S11: {result.best_value:.2f} dB", flush=True)
    print(f"Dimensions in mm: {result.best_parameters}", flush=True)
    return result


# SUPPORTING CHECKS - normally leave these unchanged.
def realised_feed_offset_mm(parameters):
    """Return the centre-to-feed distance actually placed on the design grid.

    With integer-mm edges, an even-length patch has its centre on a grid node:
    the requested integer-mm offset is exact. An odd-length patch has its
    centre halfway between nodes. Resolve that tie toward the left edge,
    giving an actual offset 0.5 mm larger than requested. Keep this fixed
    physical convention on finer solver meshes so refinement does not move
    the antenna. The helper's result is also saved with each S11 spectrum.
    """
    half_grid_shift_mm = 0.5 if parameters["length_mm"] % 2 else 0.0
    return parameters["feed_offset_mm"] + half_grid_shift_mm


def validate_model_settings(parameters):
    """Reject invalid geometry/settings before an expensive simulation starts."""
    if "feed_inset_mm" in parameters:
        raise ValueError(
            "Use feed_offset_mm from the patch centre, not feed_inset_mm; "
            "convert an old inset as length_mm/2 - feed_inset_mm"
        )
    for name, definition in PARAMETERS.items():
        # Optimisation enforces the declared step before calling this builder.
        # One-off simulate() checks may use an odd whole-mm geometry to study
        # centring. Validate its physical bounds here without snapping it.
        Integer(definition.lower, definition.upper, definition.unit).validate(parameters[name])
    if not 0 < CELL_SIZE_M < SUBSTRATE_HEIGHT_M:
        raise ValueError("Resolve the vertical feeding wire with at least two cells")
    if not 0 < PROBE_RADIUS_M < CELL_SIZE_M / 2:
        raise ValueError("Thin-wire probe radius must be smaller than half a cell")
    if not np.isfinite(COAX_IMPEDANCE_OHM) or COAX_IMPEDANCE_OHM <= 0:
        raise ValueError("The coax characteristic impedance must be positive and finite")
    if not np.isfinite(COAX_FILLER_PERMITTIVITY) or COAX_FILLER_PERMITTIVITY <= 0:
        raise ValueError("The coax filler permittivity must be positive and finite")
    # For a nonmagnetic TEM coax, Z0 approximately equals 60*ln(b/a)/sqrt(er).
    # b is a physical aperture radius, not an extra gprMax geometry object.
    log_outer_radius = (
        math.log(PROBE_RADIUS_M) + COAX_IMPEDANCE_OHM * math.sqrt(COAX_FILLER_PERMITTIVITY) / 60
    )
    if log_outer_radius >= math.log(CELL_SIZE_M):
        raise ValueError("The magnetic-frill aperture must remain smaller than a transverse cell")
    if not 0 < PML_THICKNESS_M < AIR_PADDING_M:
        raise ValueError("Leave air between the board and PML")
    if not np.isfinite(TIME_WINDOW_S) or TIME_WINDOW_S <= 0:
        raise ValueError("TIME_WINDOW_S must be positive and finite")
    for distance in (1e-3, BOARD_SIZE_M, SUBSTRATE_HEIGHT_M, AIR_PADDING_M, PML_THICKNESS_M):
        if not math.isclose(distance / CELL_SIZE_M, round(distance / CELL_SIZE_M), abs_tol=1e-9):
            raise ValueError(
                "The mesh must exactly divide the 1 mm design lattice and fixed dimensions"
            )
    edge_clearance_mm = parameters["length_mm"] / 2 - realised_feed_offset_mm(parameters)
    if edge_clearance_mm < 1:
        raise ValueError("The realised centre offset must leave at least 1 mm to the patch edge")
    if max(parameters["length_mm"], parameters["width_mm"]) * 1e-3 >= BOARD_SIZE_M:
        raise ValueError("The patch must fit within the ground plane")


def check_spectrum(spectrum):
    """Require usable target data, sufficient duration and decayed terminal traces.

    Interpolation between FFT bins does not improve independent resolution.
    No matching threshold is imposed: S11 near 0 dB is a legitimate bad design.
    """
    resolution = spectrum.independent_frequency_resolution_hz
    if resolution is None or not np.isfinite(resolution) or not 0 < resolution <= 10e6 * (1 + 1e-6):
        raise ValueError("Need <=10 MHz independent resolution; increase TIME_WINDOW_S")
    if np.isnan(spectrum.tail_relative_db) or spectrum.tail_relative_db > -40:
        raise ValueError("Terminal traces have not decayed below -40 dB; increase TIME_WINDOW_S")
    if not np.isfinite(spectrum.at(TARGET_FREQUENCY_HZ)):
        raise ValueError("S11 must be finite and valid at the target frequency")


if __name__ == "__main__":
    run_example()
