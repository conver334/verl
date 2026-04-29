Megatron-FSDP Example
========================

Last updated: 04/29/2026.

Introduction
------------

In this example, we run SFT and RL training with Megatron-FSDP:

- Runtime image: ``verlai/verl:vllm011.dev7``

Step 1: Prepare
--------------------

Download ``Megatron-LM`` and ``Megatron-Bridge``. The required Megatron-FSDP support has already been merged into
   ``Megatron-LM`` main
   (`<https://github.com/NVIDIA/Megatron-LM/pull/3191>`) and
   ``Megatron-Bridge`` main
   (`<https://github.com/NVIDIA-NeMo/Megatron-Bridge/pull/3512>`).

.. code:: bash

   git clone https://github.com/NVIDIA/Megatron-LM.git
   git clone https://github.com/NVIDIA-NeMo/Megatron-Bridge.git

Step 2: Run Megatron-FSDP SFT
----------------------------

Before launch, check and update key fields ``MODEL_PATH`` and ``SAVE_PATH`` in the script.

.. code:: bash

   bash examples/sft/gsm8k/run_qwen_megatron_fsdp.sh

Step 3: Run Megatron-FSDP RL
----------------------------

Before launch, check and update key fields in
``examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh``:

- ``actor_rollout_ref.model.path``: model name or local model path.
- ``train_files`` / ``test_files``: parquet paths for GSM8K and MATH.
- ``trainer.n_gpus_per_node`` and ``trainer.nnodes``: hardware topology.
- ``trainer.project_name`` and ``trainer.experiment_name``: experiment identifiers.

Then run:

.. code:: bash

   bash examples/grpo_trainer/run_qwen2-7b_math_megatron_fsdp.sh

The script launches RL training and enables Megatron-FSDP with:

- ``actor_rollout_ref.actor.megatron.use_mbridge=True``
- ``actor_rollout_ref.actor.megatron.vanilla_mbridge=False``
- ``actor_rollout_ref.actor.megatron.use_megatron_fsdp=True``
