from types import SimpleNamespace
import pytest
from jevlight.cooperation import TrajectoryRelations
from jevlight.observation import ALL_LANES, CONTROLLED_LANES, LaneObservation


def snapshot(*positions):
    nodes={}
    for vehicle, identity, lane in positions:
        obs=nodes.setdefault(identity,SimpleNamespace(intersection_id=identity,lanes={}))
        state=obs.lanes.setdefault(lane,LaneObservation(occupancy=.2))
        state.vehicles[vehicle]=1
    return list(nodes.values())


def test_matches_id_across_unobserved_travel_gap_without_topology():
    t=TrajectoryRelations(CONTROLLED_LANES)
    t.observe(snapshot(('v','a','NT')),0)
    assert t.related(('a','NT'),'downstream')==[]
    t.observe([],1)
    t.observe([],2)
    t.observe(snapshot(('v','b','ST')),3)
    assert t.related(('a','NT'),'downstream')==[('b','ST')]
    assert t.related(('b','ST'),'upstream')==[('a','NT')]
    t.observe(snapshot(('v','b','ST')),4)
    t.observe(snapshot(('v','b','ST')),4)
    assert t.matched_transitions==1
    with pytest.raises(ValueError):t.observe([],0)


def test_no_link_from_same_junction_lane_change_or_ambiguous_movement():
    t=TrajectoryRelations(ALL_LANES)
    t.observe(snapshot(('v','a','NT')),0)
    t.observe(snapshot(('v','a','NL')),1)
    t.observe(snapshot(('v','b','ST'),('v','b','SL')),2)
    t.observe(snapshot(('v','c','ET')),3)
    assert t.matched_transitions==0
    assert t.ambiguous_events>0


def test_pair_counts_accumulate_and_five_percent_is_strict():
    t=TrajectoryRelations(CONTROLLED_LANES)
    origin=[(str(i),'a','NT') for i in range(20)]
    destinations=[(str(i),'b','ST') for i in range(19)]+[('19','c','ET')]
    t.observe(snapshot(*origin),0);t.observe(snapshot(*destinations),1)
    assert t.downstream_counts[('a','NT')][('c','ET')]==1
    assert t.related(('a','NT'),'downstream')==[('b','ST')]
    assert t.related(('c','ET'),'upstream')==[('a','NT')]


def test_right_turn_policy_is_explicit_and_untracked_arrival_clears_mark():
    t=TrajectoryRelations(CONTROLLED_LANES)
    t.observe(snapshot(('v','a','NT')),0)
    t.observe(snapshot(('v','b','SR')),1)
    t.observe(snapshot(('v','c','ET')),2)
    assert t.matched_transitions==0
    t=TrajectoryRelations(ALL_LANES)
    t.observe(snapshot(('v','a','NT')),0)
    t.observe(snapshot(('v','b','SR')),1)
    assert t.related(('a','NT'),'downstream')==[('b','SR')]


def test_view_is_center_lane_based_current_only_and_empty_policy_explicit():
    t=TrajectoryRelations(CONTROLLED_LANES)
    t.observe(snapshot(('v','a','NT')),0);t.observe(snapshot(('v','b','ST')),1)
    center=snapshot(('current','a','NT'))[0]
    states={'b':{'current_phase':'NTST','lanes':{'ST':{'queue':0,'occupancy':0}},
                 'local_phase_priority':[123],'lane_relations':{'unrelated':1}}}
    view=t.view(center,states,include_empty_neighbours=True)
    entry=view['NT']['downstream'][0]
    assert entry=={'intersection_id':'b','lane':'ST','current_phase':'NTST','state':{'queue':0,'occupancy':0}}
    assert t.view(center,states,include_empty_neighbours=False)['NT']['downstream']==[]
    entry['state']['queue']=99
    assert states['b']['lanes']['ST']['queue']==0
