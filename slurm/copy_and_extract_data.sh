#!/bin/bash

source_dir="/lustre06/project/def-ttt/become/benthicnet-compiled/compiled_labelled_512px/tar"
destination_dir="$SLURM_TMPDIR"

start_time=$(date +%s.%N)

find "$source_dir" -type f -name "*.tar" | grep -i "\.tar$" | while IFS= read -r file; do
  cp "$file" "$destination_dir"
  tar -xf "${destination_dir}/$(basename "$file")" -C "$destination_dir"
  rm "${destination_dir}/$(basename "$file")"
done

end_time=$(date +%s.%N)

time_diff=$(echo "$end_time - $start_time" | bc)
echo "Time to copy and extract data: $time_diff seconds"
