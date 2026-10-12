"""Higher geometric symmetry reduces to the subgroup preserved by Yee samples."""
from types import SimpleNamespace

import numpy as np
import pytest

from gprMax.eigenmode_tracking import align_groups
from gprMax.sources import EigenmodeSource
from testing.validation.eigenmode_tracking.symmetry.polygon_symmetry_breaking import polygon_solver, inspect, rotation_action, mirror_u
from testing.validation.validate_fdfd_eigenmodes import _configured_solver_constants
import gprMax.config as config


@pytest.mark.parametrize('sides',[6,8])
def test_polygon_discretization_retains_only_compatible_symmetry(sides):
    with _configured_solver_constants():
        result=inspect(polygon_solver(sides,24),sides)
    assert result['c4v_material_check']==(sides==8)
    assert result['maximum_Maxwell_residual']<1e-10
    for pair in result['pairs']:
        assert pair['actions']['mirror_u']['closure']<1e-10
        assert pair['actions']['mirror_v']['closure']<1e-10
    if sides==6:
        assert all(p['gap_neff']>1e-4 for p in result['pairs'])
    else:
        assert result['pairs'][0]['gap_neff']<1e-10
        assert result['pairs'][1]['gap_neff']>1e-4
        assert all(p['actions']['native_90']['closure']<1e-10 for p in result['pairs'])
        np.testing.assert_allclose(result['pairs'][1]['actions']['native_90']['eigenvalues_real'],[-1,-1],atol=1e-10)


def test_interpolated_45_degree_rotation_is_not_accepted_as_exact_symmetry():
    with _configured_solver_constants():
        solver=polygon_solver(8,24)
        owner=EigenmodeSource(None)
        owner.normal_axis=2; owner.normal='z'; owner.direction='+'; owner.transverse_axes=(0,1)
        owner.degenerate=((1,2),); owner.mode_polarizations={}
        owner._tracking_impedance=config.sim_config.em_consts['z0']
        fields=[owner._fields_from_solver_mode(solver,m) for m in (1,2)]
        e=[[f[0] for f in fields]]; h=[[f[1] for f in fields]]
        grid=SimpleNamespace(dl=np.full(3,solver.du))
        align_groups(owner,grid,[20e9],[solver],(1,2),e,h,np.ones((1,2),bool))
        from gprMax.eigenmode_symmetry import _align_symmetry_subspaces
        with pytest.raises(ValueError,match='does not preserve'):
            _align_symmetry_subspaces(owner,grid,[20e9],[solver],(1,2),e,h,
                                     symmetry={'rotation_45':rotation_action(np.pi/4),'mirror_u':mirror_u})
