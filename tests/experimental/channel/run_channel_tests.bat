@echo off
REM Channel unit tests (conda env ``verl``).
REM NPY_DISABLE_CPU_FEATURES must be set before Python starts (avoid NumPy MKL FPE abort on some Windows CPUs).
REM Usage: double-click or: cmd /c tests\experimental\channel\run_channel_tests.bat
call conda activate verl
cd /d %~dp0..\..\..
set NPY_DISABLE_CPU_FEATURES=1
python -m pytest tests/experimental/channel/test_channel_topology.py tests/experimental/channel/test_checkpoint_control_path.py tests/experimental/channel/test_channel_worker_training.py -q --tb=short %*
