import numpy as np
import pytest

from inhand import frames
from inhand.config import Config
from inhand.env import CylinderEnv, Signals, reward_fn
from inhand.observations import DEPLOY, PRIV
from inhand.rl.buffer import Rollout
from inhand.rl.ppo import PPOConfig, make_ppo
from inhand.rl.relabel import hindsight_relabel
from inhand.scene import Scene
from inhand.tasks.monolithic import FullTaskEnv
from inhand.tasks.primitives import HOLD, apply_primitive
from inhand.planc.planner import plan
from inhand.vec import VecEnv


@pytest.fixture(scope="module")
def scene():
    return Scene(Config())


def test_axis_encoding_is_sign_invariant():
    a = frames.random_unit(np.random.default_rng(0))
    assert np.allclose(frames.axis_outer(a), frames.axis_outer(-a))


def test_errors_ignore_flip_and_roll():
    a = np.array([0.0, 0.0, 1.0])
    epos, eang = frames.errors(np.zeros(3), a, np.zeros(3), -a)
    assert epos == 0 and eang < 1e-6


def test_target_hidden_until_lift(scene):
    env = CylinderEnv(scene, Config(), mode="ground", seed=0)
    obs = env.reset()
    assert obs["deploy"][DEPLOY["target"]].sum() == 0
    assert np.abs(obs["priv"][PRIV["true_target"]]).sum() > 0  # critic sees it
    assert not env.ep.lifted


def test_inhand_reset_reveals_target(scene):
    env = CylinderEnv(scene, Config(), mode="inhand", seed=0)
    obs = env.reset()
    assert env.ep.lifted and obs["deploy"][DEPLOY["target"]][-1] == 1.0


def test_prelift_reward_is_target_free():
    w = Config().reward
    s = dict(in_tol=False, lifted=False, lifted_now=False, dropped=False, violation=False,
             success=False, action_sq=0.1, palm_obj_dist=0.2)
    assert reward_fn(w, Signals(epos=0.0, eang=0.0, **s)) == reward_fn(w, Signals(epos=1.0, eang=1.0, **s))


def test_vec_step_shapes(scene):
    vec = VecEnv([FullTaskEnv(scene, Config(), seed=i) for i in range(3)])
    obs = vec.reset()
    obs, r, d, infos = vec.step(np.zeros((3, 23)))
    assert obs["deploy"].shape == (3, DEPLOY.dim) and r.shape == (3,) and len(infos) == 3


def test_relabel_sets_target_blocks(scene):
    vec = VecEnv([FullTaskEnv(scene, Config(), seed=i) for i in range(2)])
    ppo = make_ppo(23, PPOConfig(T=8, minibatches=1), recurrent=True, device="cpu")
    ro, *_ = ppo.collect(vec, vec.reset(), ppo.actor.init_hidden(2, "cpu"))
    ro.hand_only[-1, 0] = 1
    ro.signals[:, 0, 3] = 1  # pretend lifted throughout
    extra = hindsight_relabel(ro, Config().reward, Config().success, max_rows=2)
    assert extra is not None and extra.N == 1 and extra.relabeled.all()
    assert extra.obs_d[-1, 0, DEPLOY["target"]][-1] == 1.0
    ppo.update(Rollout.concat(ro, extra))


def test_ppo_updates_both_actor_kinds(scene):
    from inhand.tasks.reorient import ReorientEnv
    for key, rec in (("priv", False), ("deploy", True)):
        vec = VecEnv([ReorientEnv(scene, Config(), seed=i) for i in range(2)])
        ppo = make_ppo(23, PPOConfig(T=8, minibatches=1), rec, "cpu", key)
        ro, *_ = ppo.collect(vec, vec.reset(), ppo.actor.init_hidden(2, "cpu"))
        st = ppo.update(ro)
        assert np.isfinite(st["pi_loss"])


def test_planner_holds_at_goal_and_moves_otherwise():
    s = Config().success
    p, a = np.array([0.02, 0.0, 0.03]), np.array([1.0, 0.0, 0.0])
    assert plan(p, a, 0.02, 0.06, p, a, s) == HOLD
    p2, a2 = apply_primitive(2, p, a, 0.02, 0.06)
    k = plan(p, a, 0.02, 0.06, p2, a2, s, depth=1)
    assert k == 2


def test_deployed_actor_rejects_privileged(scene, tmp_path):
    from inhand.rl.ppo import load_actor
    ppo = make_ppo(23, PPOConfig(), False, "cpu", "priv")
    ppo.save(str(tmp_path / "t.pt"))
    with pytest.raises(ValueError):
        load_actor(str(tmp_path / "t.pt"), deploy_only=True)
