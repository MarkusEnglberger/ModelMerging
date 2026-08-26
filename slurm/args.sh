#!/bin/bash
# Job parameters arrive as positional KEY=VALUE arguments, not via the
# environment.
#
# Slurm on Snellius does not propagate the submitting shell's environment to
# the job: `ARMS=apr sbatch job.sbatch` arrives in the job with ARMS unset, so
# `: "${ARMS:=default}"` takes the default. Positional arguments are always
# forwarded (verified with job 26079897). Use those:
#
#   sbatch slurm/cv_protocol_glue8.sbatch ARMS=apr SUFFIX=_aprS100
#
# Quote values containing spaces:
#
#   sbatch slurm/cv_protocol_glue8.sbatch "ARMS=apr nogate"
#
# Source this AFTER slurm/common.sh and BEFORE the `: "${VAR:=default}"` block.

for _kv in "$@"; do
  case "$_kv" in
    *=*) export "${_kv%%=*}"="${_kv#*=}" ;;
    *)   echo "[warn] ignoring argument that is not KEY=VALUE: $_kv" >&2 ;;
  esac
done
unset _kv

# Move an existing output file aside instead of overwriting it in place.
guard_output() {
  local path="$1"
  if [ -e "$path" ]; then
    local bak="${path%.json}.superseded-$(date +%Y%m%d-%H%M%S).json"
    echo "[guard] ${path} exists; moving it to ${bak}" >&2
    mv -- "$path" "$bak"
  fi
}
