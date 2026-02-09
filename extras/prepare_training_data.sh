#!/bin/bash
# script pour préparer toutes les images Mask pour le training

SRC_DIR="images"
DST_DIR="images_transformed"

# créer le dossier de destination
mkdir -p "$DST_DIR"

# pour chaque classe dans le dossier source
for class_dir in "$SRC_DIR"/*/; do
    class_name=$(basename "$class_dir")
    echo "Processing class: $class_name"
    
    # créer le sous-dossier pour cette classe
    mkdir -p "$DST_DIR/$class_name"
    
    # transformer les images avec l'option --mask (feuille sur fond noir)
    python Transformation.py -src "$class_dir" -dst "$DST_DIR/$class_name" --mask
    
    echo "Done: $class_name"
    echo "---"
done

echo "All classes processed!"
echo "Training data ready in: $DST_DIR"
echo ""
echo "Next steps:"
echo "1. Train: python Train.py $DST_DIR"
echo "2. Predict: python Predict.py <model_dir> <image> --mask"
