MODEL="images_model.zip"
IMAGES="images"
TYPES=(Apple_Black_rot Apple_healthy Apple_rust Apple_scab Grape_Black_rot Grape_Esca Grape_healthy Grape_spot)

for i in ${TYPES}; do
  type="${TYPES[$i]}"
  n=$([ "$i" -lt 4 ] && echo 13 || echo 12)
  while IFS= read - r image; do
    [ -n "$img" ] && python predict.py "$MODEL" "$image"
  done < <(find "$IMAGES/$type" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) 2>/dev/null | shuf -n "$n")
done
