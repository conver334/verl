# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .utils import TensorLoRARequest, VLLMHijack, is_version_ge

# The contents of vllm/patch.py should not be imported here, because the contents of
# patch.py should be imported after the vllm LLM instance is created. Therefore,
# wait until you actually start using it before importing the contents of
# patch.py separately.

__all__ = [
    "TensorLoRARequest",
    "VLLMHijack",
    "is_version_ge",
]
