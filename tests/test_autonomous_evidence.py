import hashlib
import json
from types import SimpleNamespace

import pytest

from pathmnist.autonomous import AIScientistExperimentRunner
from pathmnist.autonomous_evidence import snapshot_evidence, verified_metrics
from pathmnist.scientific_integrity import IntegrityError, record_trusted_evaluation
from pathmnist.tuning_evidence import model_signature, select_verified_tuning, validate_tuning_record


@pytest.mark.parametrize('final,search,valid', [(12,[5,5,5],True), (6,[8,8,8],False),
                                              (12,[8,8,8],False), (12,[],True)])
def test_comparison_reserves_final_training_before_search(final, search, valid):
    from pathmnist.comparison_policy import validate_final_plan
    from pathmnist.stage_policy import POLICIES
    policy = {'max_epochs':12, 'early_stopping':{'enabled':False}}
    code = f'FINAL_TRAINING_PLAN = {dict(policy, max_epochs=final, search_epochs=search)!r}'
    if valid:
        validate_final_plan(code,policy,POLICIES[3].budget)
    else:
        with pytest.raises(IntegrityError):
            validate_final_plan(code,policy,POLICIES[3].budget)


def test_comparison_allows_actual_early_stop_not_changed_cap():
    from pathmnist.comparison_policy import validate_final_manifest
    policy = {'max_epochs':12,'early_stopping':{'enabled':False}}
    validate_final_manifest(dict(policy,epochs=11),policy)
    with pytest.raises(IntegrityError):
        validate_final_manifest(dict(policy,max_epochs=6,epochs=6),policy)


def test_comparison_rejects_correct_declaration_with_wrong_final_assignment():
    from pathmnist.comparison_policy import validate_final_plan
    from pathmnist.stage_policy import POLICIES
    policy = {'max_epochs':12,'early_stopping':{'enabled':False}}
    code = f'FINAL_TRAINING_PLAN = {dict(policy,search_epochs=[5,5,5])!r}\nfinal_train_epochs = 6\n'
    with pytest.raises(IntegrityError,match='assignment'):
        validate_final_plan(code,policy,POLICIES[3].budget)


def test_wrong_comparison_plan_is_rejected_before_runner(tmp_path):
    from pathlib import Path
    class Runner:
        def run_python(self, *args, **kwargs):
            pytest.fail('Invalid final policy must not execute training')
    root = tmp_path
    (root/'research').mkdir()
    policy = {'max_epochs':12,'early_stopping':{'enabled':False}}
    (root/'research/comparison_training_policy.json').write_text(json.dumps({'policy':policy}))
    project = Path(__file__).resolve().parents[1]
    adapter = AIScientistExperimentRunner(project,object(),Runner())
    *_, cls = adapter._runtime_classes(root/'dataset/research_view',intervention_signals=('label_smoothing',))
    worker = cls(root/'worker')
    worker.stage_name = '3_creative_research_1_first_attempt'
    code = "criterion=nn.CrossEntropyLoss(label_smoothing=0.1)\nloss.backward()\n"
    code += f'FINAL_TRAINING_PLAN = {dict(policy,max_epochs=6,search_epochs=[8,8,8])!r}\n'
    result = worker.run(code)
    assert 'Final training policy must equal trusted baseline' in str(result.term_out)


@pytest.mark.parametrize('stage,required', [('3_creative_research_1_first_attempt', True),
                                         ('2_baseline_tuning_1_first_attempt', False)])
def test_dynamic_loss_runtime_guard_is_owned_by_approved_intervention(tmp_path, stage, required):
    from pathlib import Path
    captured = {}
    class Runner:
        def run_python(self, *args, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(succeeded=False, exit_code=1, timed_out=False,
                                   stdout='', stderr='offline execution substitute', elapsed_seconds=0)
    project = Path(__file__).resolve().parents[1]
    adapter = AIScientistExperimentRunner(project, object(), Runner())
    *_, interpreter_class = adapter._runtime_classes(tmp_path / 'view', intervention_signals=('label_smoothing',))
    interpreter = interpreter_class(tmp_path / 'worker')
    interpreter.stage_name = stage
    code = (project / 'tests/fixtures/semantic_recovery/proposed_metadata.py').read_text(encoding='utf-8')
    if not required:
        code = code.replace('label_smoothing=smoothing_factor', 'label_smoothing=0.0')
    interpreter.run(code)
    assert captured, 'Preflight did not reach the runner'
    assert captured.get('require_standard_smoothing', False) is required


def test_model_signature_ignores_duplicate_identical_constructors():
    baseline = "class Net(nn.Module):\n    pass\nmodel = Net(width=32)\n"
    assert model_signature(baseline) == model_signature(baseline + "restored = Net(width=32)\n")
    assert model_signature(baseline) != model_signature(baseline + "changed = Net(width=64)\n")
    assert model_signature(baseline) != model_signature(baseline.replace('pass', 'depth = 3'))


def test_model_signature_ignores_documentation_only_changes():
    baseline = 'class Net(nn.Module):\n    def forward(self, x):\n        return x\nmodel = Net()\n'
    documented = baseline.replace('class Net(nn.Module):\n', 'class Net(nn.Module):\n    """Documentation."""\n')
    assert model_signature(baseline) == model_signature(documented)


def tuning_record():
    return {"schema_version": 1, "complete": True, "seed": 0,
            "selection_metric": "accuracy", "selected_learning_rate": 0.001,
            "candidates": [{"learning_rate": lr, "validation_metric": 0.5,
                            "history": [{"epoch": 1, "train_loss": 1.0,
                                         "validation_loss": 1.0, "validation_metric": 0.5}]}
                           for lr in (0.001, 0.0003)]}


def tuning_journal(tmp_path, *, legacy=False, changed_model=False, manifests=None, record=None):
    profile, source, _, _ = evidence(tmp_path)
    view = tmp_path / "dataset/research_view"
    view.mkdir(parents=True)
    (view / "dataset_profile.json").write_bytes(profile.read_bytes())
    nodes = []
    for index in range(2):
        code = "class Net(nn.Module):\n    pass\nmodel = Net()\n" + f"# candidate {index}\n"
        if changed_model and index:
            code = code.replace("Net()", "Net(width=32)")
        digest = hashlib.sha256(code.encode()).hexdigest()
        result = json.loads((source / "experiment_result.json").read_text())
        result["code_sha256"] = digest
        (source / "experiment_result.json").write_text(json.dumps(result))
        record_trusted_evaluation(profile_path=profile, split="validation", sample_ids=["v0", "v1"],
                                  targets=[0, 1], predictions=[0, 0], probabilities=None,
                                  code_sha256=digest, output_dir=source)
        if index and not legacy:
            (source / "tuning_evidence.json").write_text(json.dumps(record or tuning_record()))
        if manifests:
            manifest = json.loads((source / "experiment_manifest.json").read_text())
            manifest.update(manifests[index])
            (source / "experiment_manifest.json").write_text(json.dumps(manifest))
        snapshot_evidence(source, tmp_path / "experiment_logs/evidence" / digest)
        nodes.append(SimpleNamespace(code=code, id=str(index), is_buggy=False,
                                     is_seed_node=False, is_seed_agg_node=False))
    return SimpleNamespace(nodes=nodes)


def stopped_manifest(epochs, maximum=15):
    return {"epochs": epochs, "max_epochs": maximum,
            "early_stopping": {"enabled": True, "monitor": "validation_metric",
                               "mode": "max", "patience": 5, "min_delta": 0.0}}


def stopped_record(counts):
    record = tuning_record()
    for row, count in zip(record["candidates"], counts):
        row["history"] = [{"epoch": epoch, "train_loss": 1.0,
                           "validation_loss": 1.0, "validation_metric": 0.5}
                          for epoch in range(1, count + 1)]
    return record


def test_legacy_loss_checkpoint_can_tune_by_primary_accuracy(tmp_path):
    baseline = stopped_manifest(2)
    baseline.update(selection_metric='validation_loss')
    baseline['early_stopping'].update(monitor='validation_loss', mode='min')
    candidate = {**baseline, 'selection_metric': 'accuracy', 'primary_metric': 'accuracy',
                 'checkpoint_selection': {'metric': 'validation_loss', 'mode': 'min'}}
    record = tuning_record()
    for row in record['candidates']:
        row['selected_epoch'] = 2
        row['history'] = [
            {'epoch': 1, 'train_loss': 1., 'validation_loss': 1., 'validation_metric': .8},
            {'epoch': 2, 'train_loss': .3, 'validation_loss': .2, 'validation_metric': .5},
        ]
    journal = tuning_journal(tmp_path, manifests=[baseline, candidate], record=record)
    node, errors = select_verified_tuning(journal, tmp_path, 'accuracy')
    assert node is journal.nodes[1], errors
    # The baseline's historical meaning was not rewritten to accuracy.
    digest = hashlib.sha256(journal.nodes[0].code.encode()).hexdigest()
    saved = json.loads((tmp_path / 'experiment_logs/evidence' / digest / 'experiment_manifest.json').read_text())
    assert saved['selection_metric'] == 'validation_loss'
    record['candidates'][0]['selected_epoch'] = 1
    with pytest.raises(IntegrityError, match='checkpoint_selection'):
        validate_tuning_record(record, {**candidate, 'seed': 0, 'learning_rate': .001}, .5, primary='accuracy')


def test_ambiguous_metric_has_actionable_diagnostic():
    from pathmnist.experiment_manifest import metric_policy, ManifestError
    with pytest.raises(ManifestError, match='Ambiguous legacy selection_metric'):
        metric_policy({'selection_metric': 'validation_metric'}, 'accuracy')


def test_raw_execution_survives_contract_failure(tmp_path):
    from pathmnist.autonomous_evidence import preserve_unvalidated_execution
    source = tmp_path / 'worker'
    source.mkdir()
    (source / 'experiment_manifest.json').write_text('{"epochs": null}')
    (source / 'model_checkpoint.pt').write_bytes(b'completed training')
    output = preserve_unvalidated_execution(source, tmp_path / 'raw', 'print(1)')
    (source / 'model_checkpoint.pt').write_bytes(b'next run')
    assert (output / 'model_checkpoint.pt').read_bytes() == b'completed training'
    assert json.loads((output / 'raw_receipt.json').read_text())['status'] == 'unvalidated'
    assert not (output / 'artifact_hashes.json').exists()


def test_rejected_nodes_keep_diagnostic_in_final_summary(tmp_path):
    journal = tuning_journal(tmp_path)
    journal.nodes[1].is_buggy = True
    journal.nodes[1].analysis = 'Baseline checkpoint selection policy changed'
    node, errors = select_verified_tuning(journal, tmp_path, 'accuracy')
    assert node is None
    assert 'checkpoint selection policy changed' in str(errors)


@pytest.mark.parametrize("counts", [(15, 14), (14, 15)])
def test_tuning_compares_limits_not_actual_epochs(tmp_path, counts):
    journal = tuning_journal(tmp_path, manifests=[stopped_manifest(14), stopped_manifest(counts[0])],
                             record=stopped_record(counts))
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is journal.nodes[1]
    assert not errors


@pytest.mark.parametrize("field,value", [("max_epochs", 16),
    ("early_stopping", {"enabled": False}),
    ("patience", 3), ("min_delta", 0.01), ("mode", "min"), ("monitor", "validation_loss")])
def test_tuning_rejects_changed_training_policy(tmp_path, field, value):
    candidate = stopped_manifest(14)
    if field in ("max_epochs", "early_stopping"):
        candidate[field] = value
    else:
        candidate["early_stopping"][field] = value
    journal = tuning_journal(tmp_path, manifests=[stopped_manifest(14), candidate],
                             record=stopped_record((14, 14)))
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is None
    assert "training policy changed" in str(errors)


def test_legacy_epoch_mismatch_is_not_reinterpreted(tmp_path):
    journal = tuning_journal(tmp_path, manifests=[{"epochs": 14}, {"epochs": 15}])
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is None
    assert "legacy evidence lacks training policy" in str(errors)


def test_mixed_legacy_and_explicit_policy_blocks(tmp_path):
    journal = tuning_journal(tmp_path, manifests=[{"epochs": 14}, stopped_manifest(14)])
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is None
    assert "policy changed or missing" in str(errors)


@pytest.mark.parametrize("counts,completed,reason", [((16, 14), 14, "excessive"),
    ((14, 15), 15, "Selected completed epochs")])
def test_explicit_policy_checks_candidate_history(counts, completed, reason):
    manifest = dict(stopped_manifest(completed), seed=0, learning_rate=0.001,
                    selection_metric="accuracy")
    with pytest.raises(IntegrityError, match=reason):
        validate_tuning_record(stopped_record(counts), manifest, 0.5)


def test_disabled_early_stopping_requires_complete_history():
    manifest = dict(stopped_manifest(14), seed=0, learning_rate=0.001,
                    selection_metric="accuracy", early_stopping={"enabled": False})
    with pytest.raises(IntegrityError, match="Incomplete training"):
        validate_tuning_record(stopped_record((14, 15)), manifest, 0.5)


@pytest.mark.parametrize("changes", [{"max_epochs": True}, {"max_epochs": 0},
    {"epochs": 16}, {"epochs": False}, {"early_stopping": None},
    {"early_stopping": {"enabled": False, "patience": 2}},
    {"early_stopping": {"enabled": True, "monitor": "validation_loss", "mode": "min",
                        "patience": 3, "min_delta": float("nan")}}])
def test_manifest_loader_validates_training_policy(tmp_path, changes):
    from pathmnist.experiment_manifest import ManifestError, load_manifest
    _, source, _, _ = evidence(tmp_path)
    path = source / "experiment_manifest.json"
    manifest = json.loads(path.read_text())
    manifest.update(stopped_manifest(14))
    manifest.update(changes)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ManifestError):
        load_manifest(path)


def test_tuning_accepts_verified_no_improvement(tmp_path):
    journal = tuning_journal(tmp_path)
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is journal.nodes[1]
    assert not errors


def test_new_execution_requires_policy_but_legacy_read_is_supported(tmp_path):
    from pathmnist.experiment_manifest import ManifestError, load_manifest
    _, source, _, _ = evidence(tmp_path)
    path = source / "experiment_manifest.json"
    assert load_manifest(path)["epochs"] == 1
    with pytest.raises(ManifestError, match="New executions require"):
        load_manifest(path, require_training_policy=True)


@pytest.mark.parametrize("option,reason", [("legacy", "Missing immutable"),
                                           ("changed_model", "architecture changed")])
def test_tuning_rejects_legacy_or_changed_model(tmp_path, option, reason):
    journal = tuning_journal(tmp_path, **{option: True})
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is None
    assert reason in str(errors)


def test_tuning_snapshot_detects_history_tampering(tmp_path):
    journal = tuning_journal(tmp_path)
    digest = hashlib.sha256(journal.nodes[1].code.encode()).hexdigest()
    saved = tmp_path / "experiment_logs/evidence" / digest / "tuning_evidence.json"
    saved.write_text("{}")
    node, errors = select_verified_tuning(journal, tmp_path, "accuracy")
    assert node is None
    assert "hash mismatch" in str(errors)


@pytest.mark.parametrize("field,value", [("complete", False), ("candidates", []),
                                        ("candidates", [None, None]), ("seed", 1),
                                        ("selected_learning_rate", 0.1)])
def test_tuning_rejects_incomplete_records(field, value):
    record = tuning_record()
    record[field] = value
    with pytest.raises(IntegrityError):
        validate_tuning_record(record, {"seed": 0, "epochs": 1, "learning_rate": 0.001,
                                        "selection_metric": "accuracy"}, 0.5)


@pytest.mark.parametrize("mutation", ["missing_history", "wrong_epoch", "wrong_score", "duplicate_lr"])
def test_tuning_rejects_inconsistent_histories(mutation):
    record = tuning_record()
    if mutation == "missing_history":
        record["candidates"][0]["history"] = []
    elif mutation == "wrong_epoch":
        record["candidates"][0]["history"][0]["epoch"] = 2
    elif mutation == "wrong_score":
        record["candidates"][0]["validation_metric"] = 0.9
    else:
        record["candidates"][1]["learning_rate"] = 0.001
    with pytest.raises(IntegrityError):
        validate_tuning_record(record, {"seed": 0, "epochs": 1, "learning_rate": 0.001,
                                        "selection_metric": "accuracy"}, 0.5)


def test_stage_limit_without_valid_candidate_never_passes(project_root, tmp_path):
    from pathmnist.autonomous import AutonomousExperimentError
    journal = tuning_journal(tmp_path)
    journal.nodes[1].is_buggy = True
    runner = AIScientistExperimentRunner(project_root, object(), object())
    manager_class, *_ = runner._runtime_classes(tmp_path / "dataset/research_view")
    manager = object.__new__(manager_class)
    stage = SimpleNamespace(name="2_baseline_tuning_1_learning_rate", max_iterations=3)
    manager.journals = {stage.name: journal}
    manager.cfg = SimpleNamespace(agent=SimpleNamespace(contract_metric="accuracy"))
    assert manager._check_stage_completion(stage)[0] is False
    stage.max_iterations = 2
    with pytest.raises(AutonomousExperimentError, match="TUNING_EVIDENCE_BLOCKED"):
        manager._check_substage_completion(stage, journal)
    stage.max_iterations = 20
    journal.nodes[1].term_out = 'ARTIFACT_REVIEW_REQUIRED: epochs metadata missing'
    with pytest.raises(AutonomousExperimentError, match='inspect and revalidate artifacts'):
        manager._check_stage_completion(stage)
    journal.nodes[1].term_out = 'Proposed-method code needs review for unknown components: novel_policy'
    with pytest.raises(AutonomousExperimentError, match='SEMANTIC_REVIEW_REQUIRED'):
        manager._check_stage_completion(stage)


def test_progress_writer_keeps_acceptance_separate(tmp_path):
    from pathmnist.autonomous import _write_agent_progress
    stage = SimpleNamespace(name="2_baseline_tuning")
    journal = SimpleNamespace(nodes=[object()], good_nodes=[])
    _write_agent_progress(tmp_path, stage, journal, [])
    progress = json.loads((tmp_path / "agent_progress.json").read_text())
    assert progress["stage"] == stage.name
    assert progress["node_count"] == 1
    assert progress["completed_agent_stages"] == []
    assert not (tmp_path / "task.json").exists()


def test_tuning_stage_and_substage_use_host_evidence(project_root, tmp_path):
    journal = tuning_journal(tmp_path)
    runner = AIScientistExperimentRunner(project_root, object(), object())
    manager_class, *_ = runner._runtime_classes(tmp_path / "dataset/research_view")
    manager = object.__new__(manager_class)
    stage = SimpleNamespace(name="2_baseline_tuning_1_learning_rate", max_iterations=3)
    manager.journals = {stage.name: journal}
    manager.cfg = SimpleNamespace(agent=SimpleNamespace(contract_metric="accuracy"))
    assert manager._check_stage_completion(stage)[0]
    assert manager._check_substage_completion(stage, journal)[0]
    assert manager._get_best_implementation(stage.name).code == journal.nodes[1].code


def test_legacy_stage_blocks_without_retry_or_llm(project_root, tmp_path):
    from pathmnist.autonomous import AutonomousExperimentError
    journal = tuning_journal(tmp_path, legacy=True)
    runner = AIScientistExperimentRunner(project_root, object(), object())
    manager_class, *_ = runner._runtime_classes(tmp_path / "dataset/research_view")
    manager = object.__new__(manager_class)
    stage = SimpleNamespace(name="2_baseline_tuning_1_learning_rate", max_iterations=3)
    manager.journals = {stage.name: journal}
    manager.cfg = SimpleNamespace(agent=SimpleNamespace(contract_metric="accuracy"))
    with pytest.raises(AutonomousExperimentError, match="TUNING_EVIDENCE_BLOCKED"):
        manager._check_stage_completion(stage)


def evidence(tmp_path):
    profile = tmp_path / "dataset_profile.json"
    profile.write_text(json.dumps({"classes": ["a", "b"], "samples": [
        {"id": "v0", "label": "a", "split": "validation"},
        {"id": "v1", "label": "b", "split": "validation"},
    ]}))
    directory = tmp_path / "process/working"
    directory.mkdir(parents=True)
    code = "print('fixture')"
    digest = hashlib.sha256(code.encode()).hexdigest()
    record_trusted_evaluation(profile_path=profile, split="validation", sample_ids=["v0", "v1"],
                              targets=[0, 1], predictions=[0, 0], probabilities=None,
                              code_sha256=digest, output_dir=directory)
    (directory / "experiment_result.json").write_text(json.dumps({
        "code_sha256": digest, "seed": 0, "split": "validation", "test_data_accessed": False,
        "sample_ids": ["v0", "v1"], "targets": [0, 1], "predictions": [0, 0],
    }))
    (directory / "experiment_manifest.json").write_text(json.dumps({
        "schema_version": 1, "dataset": "fixture", "model": "tiny", "optimizer": "Adam",
        "learning_rate": 0.001, "epochs": 1, "batch_size": 2, "seed": 0,
        "input_resolutions": [8], "selection_metric": "accuracy", "hardware": "cpu",
    }))
    (directory / "contract_execution.json").write_text("{}")
    (directory / "model_checkpoint.pt").write_bytes(b"fixture-not-executable")
    return profile, directory, code, digest


def test_verified_evidence_rejects_wrong_seed_and_hash(tmp_path):
    profile, directory, _, digest = evidence(tmp_path)
    assert verified_metrics(directory, profile, digest)["accuracy"] == 0.5
    with pytest.raises(IntegrityError, match="another generated"):
        verified_metrics(directory, profile, "wrong")
    result = json.loads((directory / "experiment_result.json").read_text())
    result["seed"] = 7
    (directory / "experiment_result.json").write_text(json.dumps(result))
    with pytest.raises(IntegrityError, match="seeds disagree"):
        verified_metrics(directory, profile, digest)


def test_snapshot_detects_checkpoint_tampering(tmp_path):
    profile, directory, _, digest = evidence(tmp_path)
    saved = tmp_path / "saved"
    snapshot_evidence(directory, saved)
    assert verified_metrics(saved, profile, digest)["accuracy"] == 0.5
    (saved / "model_checkpoint.pt").write_bytes(b"changed")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        verified_metrics(saved, profile, digest)


def test_auxiliary_phases_never_execute_training_or_llm_code(project_root, tmp_path):
    profile, directory, code, _ = evidence(tmp_path)

    class NoDocker:
        def run_python(self, *args, **kwargs):
            raise AssertionError("Auxiliary phase must not launch Docker")

    runner = AIScientistExperimentRunner(project_root, object(), NoDocker())
    *_, interpreter = runner._runtime_classes(profile.parent, intervention_signals=("color_jitter", "rotation", "flip"))
    instance = interpreter(directory.parent)
    instance.stage_name = "3_creative_research"
    assert instance.trusted_metrics_for_node(code)[0]["data"][0]["final_value"] == 0.5
    assert instance.run_for_purpose("raise RuntimeError('do not execute')", purpose="plotting").exc_type is None
    with pytest.raises(ValueError):
        instance.run_for_purpose(code, purpose="unknown")
