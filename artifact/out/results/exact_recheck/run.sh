#!/bin/bash
# one Storm call per requirement product: $1 = mode (exact | sound)
cd /work
one() { f=$1; mode=$2; b=${f%.prism}
  if [ "$mode" = exact ]; then args="--exact"; else args="--sound --precision 1e-12"; fi
  s=$(date +%s.%N); timeout 3600 storm --prism $f --prop "$(cat $b.props)" $args > $b.$mode.out 2>&1; rc=$?
  echo "rc=$rc seconds=$(echo "$(date +%s.%N) - $s" | bc -l 2>/dev/null || echo ?)" >> $b.$mode.out; }
export -f one
ls c*/*.prism | xargs -P 6 -I{} bash -c "one {} $1"
echo ALL_DONE
