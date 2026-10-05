import copy
import json
from pathlib import Path
import pytest
from benchmarks.the_well.rank_ladder import _validate_protocol

ROOT=Path(__file__).resolve().parents[1]

def test_protocol_and_reference_manifest():
    protocol=json.loads((ROOT/'benchmarks/the_well/protocol.json').read_text())
    _validate_protocol(protocol)
    manifest=json.loads((ROOT/'benchmarks/the_well/reference_feature_manifest.json').read_text())
    assert protocol['rank_scales']==manifest['rank_scaling']['scales']
    assert protocol['trace']['sha256']==manifest['trace']['sha256']
    bad=copy.deepcopy(protocol);bad['repeats']['65536']=10
    with pytest.raises(ValueError):_validate_protocol(bad)
    for gpu in ['b200','h100']:
        for path in (ROOT/'results'/gpu/'the_well').glob('ladder_r*.json'):
            data=json.loads(path.read_text())
            assert data['trace']['rank_scale']==protocol['rank_scales'][str(data['rank'])]
            assert data['trace']['sha256']==protocol['trace']['sha256']
