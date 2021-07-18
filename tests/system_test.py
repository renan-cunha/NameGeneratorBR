import subprocess
import os
import filecmp


def test_system_call(tmpdir):
    subprocess.run(["make", "train_model"])
    output_path = os.path.join(tmpdir, "output.txt")
    ground_truth_path = os.path.join("tests", "truth_output.txt")
    f = open(output_path, "w")
    subprocess.call(["python", "src/models/predict_model.py", "-cs", "4",
                     "-p", "pau", "-s", "0"], stdout=f)
    assert filecmp.cmp(output_path, ground_truth_path)
