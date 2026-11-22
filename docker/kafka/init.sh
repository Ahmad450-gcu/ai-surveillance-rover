#!/bin/bash
# One-shot: create the streaming topics (auto-creation is off, so a typo can't create a new topic silently).
set -e
K=/opt/kafka/bin/kafka-topics.sh
B=kafka:9092
create() {   # name partitions retention-hours
  $K --bootstrap-server $B --create --if-not-exists --topic "$1" --partitions "$2" --replication-factor 1 \
     --config retention.ms=$(( $3 * 3600 * 1000 ))
}
create rover.pose-frames      3 24     # per-frame skeletons from the detectors (key = camera id)
create rover.pose-frames.dlq  1 168    # records that failed validation, with the reason (kept 7 days)
create rover.alerts           1 168    # alerts raised by the stream, forwarded to the backend
$K --bootstrap-server $B --list
