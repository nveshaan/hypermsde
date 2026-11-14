#!/bin/zsh
# Encoder comparison: every detector x encoder x modality, zero-shot and few-shot over K and support seeds.
# cosine_text needs the prompt embeddings: python scripts/generate_embeddings.py text_only=true
export PATH=/opt/homebrew/bin:$HOME/.local/bin:$PATH MLFLOW_DISABLE_AGENT_HINT=1
cd "$(dirname "$0")/.."
for d in cosine_text knn lof pca; do
  for mod in chest fundus mri oct; do
    echo "=== $d $mod K=0"
    uv run python scripts/run_detector.py --multirun detector=$d modality=$mod \
      encoder=CLIP,MedImageInsight,UniMedCLIP,BiomedCLIP,MedSigLIP 2>&1 | grep -E "^(chest|fundus|mri|oct)/|\[skip\]|Error"
    echo "=== $d $mod few-shot"
    uv run python scripts/run_detector.py --multirun detector=$d modality=$mod \
      encoder=CLIP,MedImageInsight,UniMedCLIP,BiomedCLIP,MedSigLIP support=1,2,4,8,16 support_seed=0,1,2,3,4 2>&1 \
      | grep -E "^(chest|fundus|mri|oct)/|\[skip\]|Error"
  done
done
echo DONE
