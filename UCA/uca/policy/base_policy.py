from typing import Dict, Tuple
import time
import torch
import torch.nn as nn
from uca.model.common.module_attr_mixin import ModuleAttrMixin
from uca.model.common.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def predict_action_timed(
        self,
        obs_dict: Dict[str, torch.Tensor],
        sync_cuda: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Run predict_action and split wall time into:
          - encoder_s: time spent in self.obs_encoder (3D / state encoder)
          - policy_s:  remaining time inside predict_action (backbone / sampling)
          - total_s:   full predict_action wall time

        Uses forward hooks so individual policy classes need no changes.
        When sync_cuda=True (default), synchronizes CUDA around timed regions
        so GPU work is attributed more accurately.
        """
        times = {"encoder_s": 0.0, "policy_s": 0.0, "total_s": 0.0}
        enc_t0 = [None]
        use_cuda = sync_cuda and torch.cuda.is_available() and self.device.type == "cuda"

        def _sync():
            if use_cuda:
                torch.cuda.synchronize(self.device)

        def pre_hook(module, inputs):
            _sync()
            enc_t0[0] = time.perf_counter()

        def post_hook(module, inputs, output):
            _sync()
            if enc_t0[0] is not None:
                times["encoder_s"] += time.perf_counter() - enc_t0[0]
                enc_t0[0] = None

        handles = []
        if hasattr(self, "obs_encoder") and isinstance(self.obs_encoder, nn.Module):
            handles.append(self.obs_encoder.register_forward_pre_hook(pre_hook))
            handles.append(self.obs_encoder.register_forward_hook(post_hook))

        try:
            _sync()
            t0 = time.perf_counter()
            result = self.predict_action(obs_dict)
            _sync()
            times["total_s"] = time.perf_counter() - t0
        finally:
            for h in handles:
                h.remove()

        times["policy_s"] = max(0.0, times["total_s"] - times["encoder_s"])
        self._last_infer_profile = times
        return result, times

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
from typing import Dict, Tuple
import time
import torch
import torch.nn as nn
from uca.model.common.module_attr_mixin import ModuleAttrMixin
from uca.model.common.normalizer import LinearNormalizer

class BasePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    def predict_action_timed(
        self,
        obs_dict: Dict[str, torch.Tensor],
        sync_cuda: bool = True,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Run predict_action and split wall time into:
          - encoder_s: time spent in self.obs_encoder (3D / state encoder)
          - policy_s:  remaining time inside predict_action (backbone / sampling)
          - total_s:   full predict_action wall time

        Uses forward hooks so individual policy classes need no changes.
        When sync_cuda=True (default), synchronizes CUDA around timed regions
        so GPU work is attributed more accurately.
        """
        times = {"encoder_s": 0.0, "policy_s": 0.0, "total_s": 0.0}
        enc_t0 = [None]
        use_cuda = sync_cuda and torch.cuda.is_available() and self.device.type == "cuda"

        def _sync():
            if use_cuda:
                torch.cuda.synchronize(self.device)

        def pre_hook(module, inputs):
            _sync()
            enc_t0[0] = time.perf_counter()

        def post_hook(module, inputs, output):
            _sync()
            if enc_t0[0] is not None:
                times["encoder_s"] += time.perf_counter() - enc_t0[0]
                enc_t0[0] = None

        handles = []
        if hasattr(self, "obs_encoder") and isinstance(self.obs_encoder, nn.Module):
            handles.append(self.obs_encoder.register_forward_pre_hook(pre_hook))
            handles.append(self.obs_encoder.register_forward_hook(post_hook))

        try:
            _sync()
            t0 = time.perf_counter()
            result = self.predict_action(obs_dict)
            _sync()
            times["total_s"] = time.perf_counter() - t0
        finally:
            for h in handles:
                h.remove()

        times["policy_s"] = max(0.0, times["total_s"] - times["encoder_s"])
        self._last_infer_profile = times
        return result, times

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()
