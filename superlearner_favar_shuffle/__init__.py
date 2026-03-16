# ================================================================
# Superlearner FAVAR-Net v2.2 (Shuffle)
# ================================================================
#
# FAVAR-Net with expanded momentum/interaction inputs (14 theory
# interactions) and stabilized OOF via shuffled inner validation
# with M=3 averaged repeats.
#
# Usage:
#   python -m superlearner_favar_shuffle.data_transform \
#       --dataset baseline --horizon 2 --group ALL
#
#   python -m superlearner_favar_shuffle.run_superlearners \
#       --dataset baseline --horizon 2 --group ALL
#
#   python superlearner_favar_shuffle/evaluate_basemodel_supermodel.py \
#       --dataset baseline --horizon 2 --group ALL
#
# ================================================================
