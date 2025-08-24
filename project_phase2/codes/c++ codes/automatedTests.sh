#!/bin/bash

SAT_JSON="main_codes/satellites.json"
EXE="./SatellitePropagator"
NUM_SEGMENTS_LIST=(8 16)
NUM_POINTS=16
TOTAL_TIME=10000000  # Fixed total time for all simulations

# Requires jq (JSON parser for bash)
if ! command -v jq &> /dev/null; then
    echo "jq is required. Install with: sudo apt-get install jq"
    exit 1
fi

sat_count=$(jq length $SAT_JSON)
for ((i=0; i<$sat_count; i++)); do
    name=$(jq -r ".[$i].name" $SAT_JSON)
    r0x=$(jq -r ".[$i].r0[0]" $SAT_JSON)
    r0y=$(jq -r ".[$i].r0[1]" $SAT_JSON)
    r0z=$(jq -r ".[$i].r0[2]" $SAT_JSON)
    v0x=$(jq -r ".[$i].v0[0]" $SAT_JSON)
    v0y=$(jq -r ".[$i].v0[1]" $SAT_JSON)
    v0z=$(jq -r ".[$i].v0[2]" $SAT_JSON)
    A=$(jq -r ".[$i].A" $SAT_JSON)
    m=$(jq -r ".[$i].m" $SAT_JSON)
    c_d=$(jq -r ".[$i].c_d" $SAT_JSON)

    for num_segments in "${NUM_SEGMENTS_LIST[@]}"; do
        echo "Running $name with segments=$num_segments, points=$NUM_POINTS"
        $EXE $r0x $r0y $r0z $v0x $v0y $v0z $A $m $c_d $num_segments $NUM_POINTS $TOTAL_TIME
    done
done