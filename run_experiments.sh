#!/bin/bash

# Run Python scripts concurrently
# python3 zero_shot_cot.py &
# python3 few_shot_one_hop.py &
# python3 few_shot_two_hop.py &
python3 cot_with_edges.py &
python3 cot_without_edges.py &

# Wait for all processes to finish
wait
