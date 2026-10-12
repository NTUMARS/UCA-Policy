from typing import Dict
from uca.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def _run_policy_step(
        policy: BasePolicy,
        obs_dict_input: Dict,
        detail_timing: bool,
    ):
        """
        Call policy once. When detail_timing is True, also return
        encoder/policy/total seconds for this step.
        """
        if detail_timing:
            action_dict, times = policy.predict_action_timed(obs_dict_input)
            return action_dict, times
        action_dict = policy.predict_action(obs_dict_input)
        return action_dict, None
from typing import Dict
from uca.policy.base_policy import BasePolicy


class BaseRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BasePolicy) -> Dict:
        raise NotImplementedError()

    @staticmethod
    def _run_policy_step(
        policy: BasePolicy,
        obs_dict_input: Dict,
        detail_timing: bool,
    ):
        """
        Call policy once. When detail_timing is True, also return
        encoder/policy/total seconds for this step.
        """
        if detail_timing:
            action_dict, times = policy.predict_action_timed(obs_dict_input)
            return action_dict, times
        action_dict = policy.predict_action(obs_dict_input)
        return action_dict, None
