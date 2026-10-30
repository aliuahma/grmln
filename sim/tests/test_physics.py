"""Contact regressions run the same field bodies used by the video suite."""
import xml.etree.ElementTree as ET
import json

import pytest

from override_sim.field import build_scene
from override_sim.verification import EXPERIMENTS, _isolated, run_scenario


@pytest.mark.parametrize('experiment', EXPERIMENTS, ids=lambda e: e.name)
def test_real_field_dynamics(experiment):
    result = run_scenario(experiment.name)
    assert result['passed'], {key: value for key, value in result['checks'].items() if not value}
    assert json.dumps(result, allow_nan=False)
    assert len(result['trajectory']) > 60
    assert result['trajectory'][-1]['time_s'] == pytest.approx(experiment.seconds, abs=.003)


def test_isolation_preserves_actual_field_colliders_and_mesh_assets():
    experiment = EXPERIMENTS[0]
    original = ET.fromstring(build_scene().xml)
    _, isolated, _, _ = _isolated(experiment)
    assert ET.tostring(original.find('asset')) == ET.tostring(isolated.find('asset'))
    for name in experiment.bodies:
        source = original.find(f'./worldbody/body[@name="{name}"]')
        actual = isolated.find(f'./worldbody/body[@name="{name}"]')
        assert ET.tostring(source) == ET.tostring(actual)
    assert ET.tostring(original.find('./worldbody/geom[@name="floor"]')) == ET.tostring(
        isolated.find('./worldbody/geom[@name="floor"]'))


def test_mechanism_guides_remain_fixed_and_actuator_is_original():
    experiment = next(e for e in EXPERIMENTS if e.name == 'loader_cycle')
    original = ET.fromstring(build_scene().xml)
    _, isolated, model, _ = _isolated(experiment)
    assert model.nv == 1
    guides = isolated.findall('./worldbody/geom')
    guides = [geom for geom in guides if geom.get('name', '').startswith('loader_guide_0_')]
    assert len(guides) == 2
    for guide in guides:
        assert ET.tostring(guide) == ET.tostring(
            original.find(f'./worldbody/geom[@name="{guide.get("name")}"]'))
    assert len(isolated.findall('./worldbody/geom[@class="display"]')) > 0
    assert ET.tostring(isolated.find('./actuator/position')) == ET.tostring(
        original.find('./actuator/position[@name="loader_0_lift"]'))


def test_unknown_scenario_is_rejected():
    with pytest.raises(ValueError, match='Unknown experiment'):
        run_scenario('toy_fixture')
