# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import OmegaConf

from verl.experimental.channel.ray_trainer import ChannelRayPPOTrainer
from verl.trainer.main_ppo import get_ppo_trainer_class
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def test_channel_mode_enabled_selects_channel_trainer():
    cfg = OmegaConf.create({"trainer": {"channel_mode": {"enabled": True}}})
    assert get_ppo_trainer_class(cfg) is ChannelRayPPOTrainer


def test_channel_mode_disabled_selects_ray_trainer():
    cfg = OmegaConf.create({"trainer": {"channel_mode": {"enabled": False}}})
    assert get_ppo_trainer_class(cfg) is RayPPOTrainer


def test_channel_mode_missing_defaults_to_ray_trainer():
    cfg = OmegaConf.create({"trainer": {}})
    assert get_ppo_trainer_class(cfg) is RayPPOTrainer
