#!/usr/bin/env bash
# Frees disk space during a Harbor run by deleting each Terminal-Bench task image
# once its trial has finished. Harbor 0.22.0 removes each trial's container but
# keeps the task image it pulled, so without this every image stays on disk.
#
# Run it in a second terminal, from the project root, while the job runs:
#   bash scripts/prune_finished_images.sh jobs/<job-name>
# Stop it with Ctrl-C.

JOB_DIR="${1:?usage: bash scripts/prune_finished_images.sh jobs/<job-name>}"

while true; do
  docker images --filter "reference=alexgshaw/*" --format "{{.Repository}}:{{.Tag}}" |
  while read -r image; do
    task="${image#alexgshaw/}"
    task="${task%%:*}"
    # Harbor writes result.json after removing the trial's container,
    # so the image is no longer in use
    if compgen -G "$JOB_DIR/${task}__*/result.json" > /dev/null; then
      docker rmi "$image" > /dev/null && echo "$(date +%H:%M) removed $image"
    fi
  done
  # Docker's current image and container usage
  echo "$(date +%H:%M) $(docker system df --format '{{.Type}} {{.Size}}' | head -2 | paste -sd ',' -)"
  sleep 60
done
