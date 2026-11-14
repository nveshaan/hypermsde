#!/bin/zsh
# Every baseline x modality, all datasets: zero-shot, then few-shot over K and support seeds.
# Log: experiments/run_baselines.log. AA-CLIP needs scripts/train_aaclip.py first; ANoCo needs DINOv3 access.
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH MLFLOW_DISABLE_AGENT_HINT=1
cd "$(dirname "$0")/.."
run() {
  uv run python scripts/run_baseline.py "$@" dataset=null 2>&1 \
    | tr '\r' '\n' | grep -E "^(chest|fundus|mri|oct)/|\[skip\]|Error|Traceback"
}
for m in anomalyclip faprompt adaclip bayespfl aaclip; do
  for mod in chest fundus mri oct; do
    echo "=== $m $mod"
    run baselines=$m modality=$mod
  done
done
for m in anoco dpmm mediclip univad anomamed; do
  for mod in chest fundus mri oct; do
    for k in 1 2 4 8 16; do
      for s in 0 1 2 3 4; do
        echo "=== $m $mod K=$k seed=$s"
        run baselines=$m modality=$mod support=$k support_seed=$s
      done
    done
  done
done
echo DONE
