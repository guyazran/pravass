"""The README's code runs as documented.

The s3e example runs only when s3e is installed, with a fake VLM in place of
the real model and a blank image in place of the photo.
"""

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"


def python_blocks():
    return re.findall(r"```python\n(.*?)```", README.read_text(), flags=re.DOTALL)


def test_examples_without_images_run():
    namespace: dict = {}
    for block in python_blocks():
        if "s3e" in block:
            continue
        exec(compile(block, str(README), "exec"), namespace)
    assert namespace["belief"].normalization.exact


def test_without_images_matches_comments():
    from pravass import MutexGroups, Normalization, TaskBelief, UnconstrainedBelief, pravass

    belief = TaskBelief(UnconstrainedBelief({"f1": 0.4, "f2": 0.7, "f3": 0.9}), MutexGroups([["f1", "f2"]]))
    assert belief.normalization == Normalization(0.72, exact=True)
    assert belief(frozenset({"f2", "f3"})) == pytest.approx(0.525)
    assert sorted(sorted(s) for s in pravass(belief, theta=0.8)) == [["f1", "f3"], ["f2", "f3"], ["f3"]]


def test_s3e_example_runs():
    pytest.importorskip("s3e")
    pytest.importorskip("unified_planning")
    from s3e.backends import VLMBackend, VLMOutput

    class FakeVLM(VLMBackend):
        """Answers "yes" with probability 0.8 for features of block a, else 0.3."""

        def query(self, images, prompt, system_prompt=None, generate=False, interest_tokens=None, **kwargs):
            yes = 0.8 if "block a" in prompt else 0.3
            probs = {"yes": yes, "no": 1 - yes}
            if interest_tokens is not None:
                probs = {t: probs.get(t, 0.0) for t in interest_tokens}
            return VLMOutput(token_probs=probs, text=None, argmax_in_interest=True)

    (block,) = [b for b in python_blocks() if "from s3e" in b]
    block = re.sub(r'vlm="[^"]*",[^\n]*', "vlm=FAKE_VLM,", block)
    block = block.replace('Image.open("scene.png")', 'Image.new("RGB", (8, 8))')
    namespace = {"FAKE_VLM": FakeVLM()}
    exec(compile(block, str(README), "exec"), namespace)
    assert namespace["constraint"].support <= namespace["belief"].features
    assert namespace["belief_set"]
    assert len(namespace["up_states"]) == len(namespace["belief_set"])
