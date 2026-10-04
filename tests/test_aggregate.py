import importlib.util
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('release_aggregate',ROOT/'benchmarks/the_well/aggregate.py')
aggregate=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregate)

@pytest.mark.parametrize('device',['b200','h100'])
def test_archived_csv(device):
    folder=ROOT/'results'/device/'the_well'
    assert len(list(folder.glob('ladder_r*.json')))==7
    assert aggregate.aggregate_csv(str(folder))==(folder/'aggregate.csv').read_bytes()
    rows=aggregate.aggregate_json(str(folder))['rows']
    assert all(row['all_correctness_passed'] for row in rows)
    if device=='h100':assert round(rows[-1]['mean_transition_speedup'],4)==71.7546
    else:
        assert round(rows[-1]['median_transition_speedup'],2)==112.27
        assert max(row['max_incremental_operator_relative_error'] for row in rows)<1.13e-4
        assert max(row['max_incremental_factor_relative_error'] for row in rows)<7.03e-6
