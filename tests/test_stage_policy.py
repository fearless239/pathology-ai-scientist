import pytest

from pathmnist.autonomous import _validate_experiment_execution_budget
from pathmnist.stage_policy import POLICIES, stage_policy, check_upstream_compatibility
from pathmnist.training_budget import training_workload
from pathmnist.scientific_integrity import IntegrityError


PROGRAM = '''
max_epochs = 15
candidate_epochs = 5
alphas = [0.05, 0.1, 0.15]
def train(alpha, epochs=None):
    loss.backward()
    optimizer.step()
for alpha in alphas:
    train(alpha, epochs=candidate_epochs)
train(selected_alpha, epochs=max_epochs)
'''


def test_subset_selection_and_final_fit_share_one_upstream_node():
    assert training_workload(PROGRAM) == (4, 30)
    _validate_experiment_execution_budget(PROGRAM, '3_creative_research_1_first_attempt')


def test_total_epoch_budget_includes_loop_multiplicity_and_final_fit():
    from pathmnist.experiment_manifest import ManifestError, validate_execution_budget
    estimate = _validate_experiment_execution_budget(PROGRAM.replace('candidate_epochs = 5','candidate_epochs = 8'), '3_creative_research')
    assert estimate['estimated_epochs'] == 39
    with pytest.raises(ManifestError, match='total epoch budget'):
        validate_execution_budget({'training_runs': [{'max_epochs': n, 'epochs': n} for n in (8, 8, 8, 15)]}, POLICIES[3].budget)


def test_dynamic_training_loop_cannot_escape_budget():
    with pytest.raises(IntegrityError, match='static bound'):
        training_workload(PROGRAM.replace('for alpha in alphas:', 'for alpha in load_candidates():'))


def test_policy_is_shared_by_roles_goals_and_prompts():
    policy = stage_policy('3_creative_research_2_any_name')
    assert policy is POLICIES[3]
    assert policy.role == 'proposed_method'
    assert '4 training launches' in policy.prompt()
    assert '30 total epochs' in policy.prompt()
    assert 'subset' in policy.goal


def test_upstream_accepts_multistep_prose_without_a_second_scheduler():
    check_upstream_compatibility({'research_question':'20% subset tuning followed by full training'})
    with pytest.raises(ValueError, match='no second scheduler'):
        check_upstream_compatibility({'execution_plan':{'steps':[]}})



def test_real_upstream_loop_uses_local_acceptance_without_vlm(project_root, tmp_path, monkeypatch):
    from types import SimpleNamespace as N
    from pathmnist.autonomous import AIScientistExperimentRunner
    runner = AIScientistExperimentRunner(project_root, object(), object())
    manager_class, *_ = runner._runtime_classes(tmp_path / 'dataset/research_view')
    from ai_scientist.treesearch.agent_manager import AgentManager
    stage = N(name='3_creative_research_1_first_attempt', stage_number=3, goals=POLICIES[3].prompt(), max_iterations=4)
    journal = N(nodes=[])
    manager = object.__new__(manager_class)
    manager.current_stage = stage
    manager.stages = [stage]
    manager.stage_history = []
    manager.journals = {stage.name: journal}
    calls=[]
    class Agent:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def step(self, callback):
            calls.append('step')
            journal.nodes.append(N(is_buggy=len(calls)==1, is_buggy_plots=None,
                                   parent=object(), is_seed_node=False, is_seed_agg_node=False,
                                   _term_out=[]))
        def _run_multi_seed_evaluation(self, node):
            calls.append('seeds')
            return []
        def _run_plot_aggregation(self, *args):
            calls.append('aggregate')
    manager._create_agent_for_stage = lambda stage: Agent()
    manager._get_best_implementation = lambda name: journal.nodes[-1]
    manager._save_checkpoint = lambda: calls.append('checkpoint')
    manager._create_next_main_stage = lambda *args: None
    def forbidden(*args, **kwargs):
        raise AssertionError('No VLM or paid model call allowed in stage acceptance')
    monkeypatch.setattr('ai_scientist.treesearch.agent_manager.query', forbidden)
    AgentManager.run(manager, forbidden)
    assert calls == ['step','step','seeds','aggregate','checkpoint']
    assert manager.current_stage is None
