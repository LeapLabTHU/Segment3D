

file_list=($(ls ../demo_test))

for scene_name in "${file_list[@]}"; do
  echo "$scene_name"
  ./segmentator ../demo_test/${scene_name}/mesh.ply 0.01 20
done


