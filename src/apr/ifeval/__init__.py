"""Official IFEval checkers, vendored from lm-evaluation-harness (tasks/ifeval),
which itself vendors google-research/instruction_following_eval. Covers all 25
verifiable instruction types (vs. the 12 hand-written approximations previously in
causal_lm.IFEVAL_CHECKERS, which restricted scoring to a biased 41% prompt subset).

Requires: immutabledict, langdetect, nltk (+ punkt/punkt_tab data; set NLTK_DATA
for offline compute nodes -- see slurm/common.sh).

`strict_follows_all` replicates the *strict* prompt-level judgement of the official
evaluation_main/lm-eval process_results: every instruction of the prompt must be
followed by the raw response (no loose-mode response rewriting).
"""

from .instructions_registry import INSTRUCTION_DICT


def strict_follows_all(prompt: str, instruction_id_list, kwargs_list, response: str) -> bool:
    """Prompt-level strict accuracy for one example."""
    if not response.strip():
        return False
    kwargs_list = kwargs_list or [{}] * len(instruction_id_list)
    for instruction_id, kw in zip(instruction_id_list, kwargs_list):
        instruction_cls = INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)
        kw = {k: v for k, v in (kw or {}).items() if v is not None}
        instruction.build_description(**kw)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)
        if not instruction.check_following(response):
            return False
    return True
