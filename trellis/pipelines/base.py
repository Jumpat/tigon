from typing import *
from contextlib import contextmanager
import torch
import torch.nn as nn
from .. import models


class Pipeline:
    """
    A base class for pipelines.
    """
    def __init__(
        self,
        models: dict[str, nn.Module] = None,
    ):
        self._offload_enabled = False
        self._active_runtime_modules: set[str] = set()
        self._offload_device = torch.device("cpu")
        self._execution_device = torch.device("cpu")
        if models is None:
            return
        self.models = models
        for model in self.models.values():
            model.eval()
        self._execution_device = self._infer_runtime_device(default=torch.device("cpu"))

    @staticmethod
    def from_pretrained(path: str) -> "Pipeline":
        """
        Load a pretrained model.
        """
        import os
        import json
        is_local = os.path.exists(f"{path}/pipeline.json")
        if is_local:
            config_file = f"{path}/pipeline.json"
        else:
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(path, "pipeline.json")

        with open(config_file, 'r') as f:
            args = json.load(f)['args']

        _models = {}
        for k, v in args['models'].items():
            try:
                _models[k] = models.from_pretrained(f"{path}/{v}")
            except Exception as e:
                print(e)
                _models[k] = models.from_pretrained(v)

        new_pipeline = Pipeline(_models)
        new_pipeline._pretrained_args = args
        return new_pipeline

    @property
    def device(self) -> torch.device:
        if getattr(self, '_execution_device', None) is not None:
            return self._execution_device
        runtime_device = self._infer_runtime_device()
        if runtime_device is not None:
            return runtime_device
        raise RuntimeError("No device found.")

    @property
    def execution_device(self) -> torch.device:
        return self._execution_device

    @property
    def offload_enabled(self) -> bool:
        return self._offload_enabled

    def _get_runtime_modules(self) -> dict[str, nn.Module]:
        return dict(getattr(self, 'models', {}))

    def _infer_runtime_device(self, default: Optional[torch.device] = None) -> Optional[torch.device]:
        for model in self._get_runtime_modules().values():
            if hasattr(model, 'device'):
                return torch.device(model.device)
        for model in self._get_runtime_modules().values():
            if hasattr(model, 'parameters'):
                try:
                    return next(model.parameters()).device
                except StopIteration:
                    continue
        return default

    def _move_runtime_modules(
        self,
        device: torch.device,
        module_names: Optional[Iterable[str]] = None,
    ) -> None:
        runtime_modules = self._get_runtime_modules()
        if module_names is None:
            module_names = runtime_modules.keys()
        for name in module_names:
            if name in runtime_modules:
                runtime_modules[name].to(device)

    def _activate_runtime_modules(self, module_names: Iterable[str]) -> None:
        if not self._offload_enabled:
            return

        runtime_modules = self._get_runtime_modules()
        requested = {name for name in module_names if name in runtime_modules}
        to_offload = self._active_runtime_modules - requested
        to_load = requested - self._active_runtime_modules

        for name in to_offload:
            runtime_modules[name].to(self._offload_device)
        if to_offload and self._execution_device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()

        for name in to_load:
            runtime_modules[name].to(self._execution_device)

        self._active_runtime_modules = requested

    @contextmanager
    def use_runtime_modules(self, module_names: Iterable[str]):
        if not self._offload_enabled:
            yield
            return

        previous = set(self._active_runtime_modules)
        self._activate_runtime_modules(module_names)
        try:
            yield
        finally:
            self._activate_runtime_modules(previous)

    def to(self, device: torch.device) -> None:
        device = torch.device(device)
        self._execution_device = device
        self._offload_enabled = False
        self._move_runtime_modules(device)
        self._active_runtime_modules = set(self._get_runtime_modules().keys())

    def enable_sequential_offload(
        self,
        execution_device: Union[str, torch.device] = "cuda",
        offload_device: Union[str, torch.device] = "cpu",
    ) -> "Pipeline":
        execution_device = torch.device(execution_device)
        offload_device = torch.device(offload_device)

        self._execution_device = execution_device
        self._offload_device = offload_device
        self._active_runtime_modules = set()

        if execution_device == offload_device or execution_device.type != 'cuda':
            self.to(execution_device)
            return self

        self._move_runtime_modules(offload_device)
        self._offload_enabled = True
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def disable_sequential_offload(self) -> "Pipeline":
        self.to(self._execution_device)
        return self

    def cuda(self) -> None:
        self.to(torch.device("cuda"))

    def cpu(self) -> None:
        self.to(torch.device("cpu"))
