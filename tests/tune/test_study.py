import numpy as np
import pytest

from skrecsys.tune import Categorical, Float, Int, Study


def _quadratic(trial):
    return -((trial.suggest_float("x", -10, 10) - 3.0) ** 2)


def test_tpe_finds_the_optimum_of_a_quadratic():
    study = Study(random_state=0)
    study.optimize(_quadratic, n_trials=60)
    assert study.best_params["x"] == pytest.approx(3.0, abs=0.2)
    assert all(study.best_value >= t.value for t in study.trials if t.value is not None)


def test_tpe_beats_random_search_on_average():
    def best(sampler, seed):
        study = Study(sampler=sampler, random_state=seed)
        study.optimize(_quadratic, n_trials=40)
        return study.best_value

    tpe = np.mean([best("tpe", seed) for seed in range(10)])
    random = np.mean([best("random", seed) for seed in range(10)])
    assert tpe > random


def test_minimize():
    study = Study(direction="minimize", random_state=0)
    study.optimize(lambda t: (t.suggest_float("x", -10, 10) - 3.0) ** 2, n_trials=60)
    assert study.best_params["x"] == pytest.approx(3.0, abs=0.2)


def test_suggestions_stay_in_their_distributions():
    def objective(trial):
        n = trial.suggest_int("n", 5, 1000, log=True)
        m = trial.suggest_int("m", -3, 3)
        lr = trial.suggest_float("lr", 1e-4, 1.0, log=True)
        kind = trial.suggest_categorical("kind", ["a", "b", None])
        return -abs(np.log(n) - 3) - abs(m) - abs(np.log(lr) + 3) - (kind != "b")

    study = Study(random_state=0)
    study.optimize(objective, n_trials=50)
    for trial in study.trials:
        p = trial.params
        assert Int(5, 1000).contains(p["n"])
        assert Int(-3, 3).contains(p["m"])
        assert Float(1e-4, 1.0).contains(p["lr"])
        assert p["kind"] in ("a", "b", None)
    assert study.best_params["kind"] == "b"
    assert {t.params["m"] for t in study.trials} == set(range(-3, 4))


def test_the_same_random_state_repeats_the_study():
    def run():
        study = Study(random_state=7)
        study.optimize(_quadratic, n_trials=25)
        return [t.params["x"] for t in study.trials]

    assert run() == run()


def test_define_by_run_spaces_may_differ_between_trials():
    def objective(trial):
        if trial.suggest_categorical("model", ["lin", "quad"]) == "lin":
            return trial.suggest_float("a", 0, 1)
        return 2 - (trial.suggest_float("b", 0, 2) - 1) ** 2

    study = Study(random_state=0)
    study.optimize(objective, n_trials=40)
    assert study.best_params["model"] == "quad"
    assert "a" not in study.best_params


def test_enqueued_params_are_used_when_inside_the_distribution():
    study = Study(random_state=0)
    study.enqueue({"x": 1.5, "n": 100})
    trial = study.ask()
    assert trial.suggest("x", Float(0, 2)) == 1.5
    assert trial.suggest("n", Int(1, 10)) != 100  # outside: sampled instead
    assert Int(1, 10).contains(trial.params["n"])


def test_repeated_suggestions_return_the_first_answer():
    trial = Study(random_state=0).ask()
    assert trial.suggest_float("x", 0, 1) == trial.suggest_float("x", 0, 1)
    with pytest.raises(ValueError, match="already suggested"):
        trial.suggest("x", Categorical([1, 2]))


def test_failed_trials_are_not_best():
    study = Study(random_state=0)
    first = study.ask()
    first.suggest_float("x", 0, 1)
    study.tell(first, float("nan"))
    with pytest.raises(ValueError, match="No trial"):
        study.best_trial  # noqa: B018
    second = study.ask()
    second.suggest_float("x", 0, 1)
    study.tell(second, 1.0)
    assert study.best_trial is second
    assert first.state == "failed"
    with pytest.raises(ValueError, match="not a running trial"):
        study.tell(second, 2.0)


def test_an_exception_fails_the_trial_and_propagates():
    study = Study(random_state=0)

    def objective(trial):
        trial.suggest_float("x", 0, 1)
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        study.optimize(objective, n_trials=3)
    assert [t.state for t in study.trials] == ["failed"]


@pytest.mark.parametrize("distribution", [(0, 1), "uniform", None, range(3)])
def test_an_unknown_distribution_raises(distribution):
    trial = Study(random_state=0).ask()
    with pytest.raises(TypeError, match="needs a Float, Int or Categorical"):
        trial.suggest("x", distribution)


def test_an_unknown_distribution_raises_even_for_an_enqueued_value():
    study = Study(random_state=0)
    study.enqueue({"x": 0.5})
    with pytest.raises(TypeError, match="needs a Float, Int or Categorical"):
        study.ask().suggest("x", (0, 1))  # ty: ignore[invalid-argument-type]


def test_the_sampler_rejects_an_unknown_distribution_directly():
    """The default branches of the sampler's matches, reached without suggest's check."""
    with pytest.raises(TypeError, match="needs a Float, Int or Categorical"):
        Study(random_state=0)._sample(0, "x", (0, 1))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "kwargs",
    [{"direction": "up"}, {"sampler": "grid"}, {"n_startup_trials": -1}, {"n_ei_candidates": 0}],
)
def test_invalid_study_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        Study(**kwargs)
