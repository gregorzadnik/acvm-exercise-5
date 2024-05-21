# acvm-exercise-5
Implementation of a long-term tracker

To run the tracker:

`python run_tracker.py --dataset . --net siamfc_net.pth --results_dir results`

To evaluate the results:

`python performance_evaluation.py --dataset . --results_dir results`

To show the tracking:

`python show_tracking.py --dataset . --results_dir results --sequence car9`