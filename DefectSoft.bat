cd /D "%~dp0"
call conda activate python37
python segmentation_tool.py
call conda deactivate