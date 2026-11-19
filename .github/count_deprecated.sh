#!/usr/bin/env bash
# Count the compiler's deprecation warnings in a build log (gcc / clang
# -Wdeprecated-declarations, nvcc "was declared deprecated") and report them
# as a warning annotation and in the job summary: they flag C++ APIs that
# PyTorch is about to remove. Python packaging notices are not counted, and
# the build does not fail on them.
# usage: count_deprecated.sh BUILD_LOG LABEL
set -euo pipefail
log=$1
label=$2
pattern='-Wdeprecated|was declared deprecated'
n=$(grep -cE -- "$pattern" "$log" || true)
echo "### $label: $n compiler deprecation warnings" >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
if [ "$n" -gt 0 ]; then
  grep -E -- "$pattern" "$log" | sort | uniq -c | sort -rn | head -20
  echo "::warning title=deprecated APIs ($label)::$n compiler deprecation warnings"
else
  echo "no deprecation warnings"
fi
