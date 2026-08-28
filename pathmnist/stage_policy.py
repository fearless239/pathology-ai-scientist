"""Local policy for the upstream four-stage search, not a second scheduler."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentExecutionBudget:
    max_conditions: int = 2
    max_total_epochs: int = 30


@dataclass(frozen=True)
class StagePolicy:
    role: str
    budget: ExperimentExecutionBudget
    goal: str

    def prompt(self):
        return (f'{self.goal} One upstream node may contain up to '
                f'{self.budget.max_conditions} training launches and '
                f'{self.budget.max_total_epochs} total epochs across all launches. '
                'Candidate count multiplied by candidate epochs plus final-training epochs '
                'must fit this total. Use only the mounted train/validation data and one '
                'host-injected seed. Select parameters on validation only; export the selected '
                'final checkpoint and predictions. Never retrain a different comparison arm. '
                'The node is complete only after all its required work succeeds. '
                'An interrupted node is not automatically a completed sub-experiment.')


POLICIES = {
    1: StagePolicy('baseline', ExperimentExecutionBudget(1, 15),
                   'Create a reproducible baseline on the sole local dataset.'),
    2: StagePolicy('baseline', ExperimentExecutionBudget(2, 30),
                   'Tune baseline learning rate, preserving architecture and all other controls.'),
    3: StagePolicy('proposed_method', ExperimentExecutionBudget(4, 30),
                   'Implement the approved intervention. When requested, perform bounded subset '
                   'parameter selection followed by full-training-set fitting within this node. '
                   'Reserve the baseline final max_epochs and unchanged early_stopping policy first; '
                   'only search may use the remaining epoch budget. Never shorten final training to fit search. '
                   'Reuse the trusted baseline rather than training it again.'),
    4: StagePolicy('component_ablation', ExperimentExecutionBudget(1, 15),
                   'Train one removed-component variant and reuse the trusted full-method result.'),
}


def stage_policy(name):
    if 'hard_routing_repair' in name:
        return POLICIES[3]
    try:
        return POLICIES.get(int(name.split('_', 1)[0]))
    except ValueError:
        return None


def enabled_stages(contract):
    policy = contract.get('experiment_policy', {})
    return [1, *([2] if policy.get('tuning', True) else []), 3,
            *([4] if policy.get('ablation', True) else [])]


def check_upstream_compatibility(contract):
    # A declarative DAG is not an upstream AgentManager task. Never silently
    # reinterpret an approved DAG; normal prose experiments remain upstream nodes.
    if contract.get('execution_plan') is not None:
        raise ValueError('Approved execution_plan requires explicit migration to upstream stages; no second scheduler is enabled')
