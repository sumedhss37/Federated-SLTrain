# Paste this into a Kaggle notebook cell after uploading this project folder.
# It verifies that the core SLTrain layer and compressor work before launching C4.

import sys, os
sys.path.insert(0, os.getcwd())

from tests.test_sl_layer import test_forward_and_backward_against_dense
from tests.test_compression import test_compression_error_feedback_shape

test_forward_and_backward_against_dense()
test_compression_error_feedback_shape()
print("Core tests passed.")
