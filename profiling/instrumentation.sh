#!/bin/bash
#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/10/24


profile_nsys() {
  if [ $# -lt 1 ]; then
    echo "Usage: profile_nsys <output_file> [additional_args...]"
    return 1
  fi

  local output_file="$1"
  shift  # remove the first parameter and use the rest for the profiling commands

  sudo nsys profile --gpu-metrics-devices all \
    --trace=nvtx,cuda -o "${output_file}" "$@"
}

nsys_to_sqlite() {
  if [ $# -lt 2 ]; then
    echo "Usage: nsys_to_sqlite <input_file> <output_file>"
    return 1
  fi

  local input_file="$1"
  local output_file="$2"

  nsys export -t sqlite "$input_file" -o "$output_file"
}