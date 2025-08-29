#!/bin/bash

SAT_JSON="satellites.json"
EXE="./build/SatellitePropagator"
NUM_SEGMENTS_LIST=(8 16 32)  
NUM_POINTS_LIST=(8 16 32)    
TOTAL_TIME=2592000  # Total time for all simulations 30 days
ABS_TOL=1e-12
REL_TOL=1e-9

# Requires jq (JSON parser for bash)
if ! command -v jq &> /dev/null; then
    echo "jq is required. Install with: sudo apt-get install jq"
    exit 1
fi

sat_count=$(jq length $SAT_JSON)
echo "got $sat_count satellites"

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

    # Nested loops: for each satellite, test all combinations of segments and points
    for num_segments in "${NUM_SEGMENTS_LIST[@]}"; do
        for num_points in "${NUM_POINTS_LIST[@]}"; do
            echo "Running $name with segments=$num_segments, points=$num_points"
            echo "command: $EXE \"$name\" $r0x $r0y $r0z $v0x $v0y $v0z $A $m $c_d $num_segments $num_points $TOTAL_TIME $ABS_TOL $REL_TOL"
            "$EXE" "$name" "$r0x" "$r0y" "$r0z" "$v0x" "$v0y" "$v0z" "$A" "$m" "$c_d" "$num_segments" "$num_points" "$TOTAL_TIME" "$ABS_TOL" "$REL_TOL"
        done
    done
    
    echo "Committing results for $name..."
    git add results/*.csv
    git commit -m "Add results for satellite: $name (segments: ${NUM_SEGMENTS_LIST[*]}, points: ${NUM_POINTS_LIST[*]})"
    git push
done
