
echo "tmp_euler_config.json                                                                                                                                                       100%  557    70.5KB/s   00:00
llm_summarization_mt.py                                                                                                                                                     100% 9880     1.2MB/s   00:00
eval_config_zgbdjpznsc.yaml                                                                                                                                                 100% 1303   193.9KB/s   00:00

The following have been reloaded with a version change:
  1) gcc/4.8.5 => gcc/8.2.0

No inference.json path provided
### Changing to project dir...
### Add to python path: /cluster/home/roysc/roysc/lm-evaluation-harness/
### Pulling commits...


The following have been reloaded with a version change:
  1) gcc/4.8.5 => gcc/8.2.0

Already up to date.
Already on 'llm_summ_mt'
M	lm_eval/tasks/llm_summarization_mt.py
Your branch is up to date with 'origin/llm_summ_mt'.

### Loading modules...

### Activating environment...

### Changing to code dir...

### Retrieving commands to execute...
tmp_euler_config.json

### Running command...
sbatch -o /cluster/home/roysc/roysc/lm-evaluation-harness/euler_outputs/principal-ampere-2023-09-07_10-11-52 --time=18:00:00 -n 4 -A ls_infk --mem-per-cpu=8G --gpus=a100_80gb:2 --wrap="python3 main.py --config  configs/eval_config_zgbdjpznsc.yaml "
Submitted batch job 27042978

"