#!/bin/zsh
# Every baseline x modality, all datasets: zero-shot, then few-shot over K and support seeds. Per-image model outputs
# of the zero-shot baselines are cached under experiments/cache (cache_dir in configs/baseline.yaml).
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
# One pass over each dataset's images per (baseline, modality): every image is encoded once and scored by all 25
# (K, seed) supports, instead of once per support.
for m in anoco dpmm mediclip univad anomamed; do
  for mod in chest fundus mri oct; do
    echo "=== $m $mod K=[1,2,4,8,16] seeds=[0..4]"
    run baselines=$m modality=$mod "supports=[1,2,4,8,16]" "support_seeds=[0,1,2,3,4]"
  done
done
echo DONE
