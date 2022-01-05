from reprod_log import ReprodLogger
from reprod_log import ReprodDiffHelper

# yolo map 指标来源：https://aistudio.baidu.com/aistudio/competition/detail/106/0/task-definition 验收标准：VOC2017 YOLO mAP:63.4 参考原论文Table1
import numpy as np
reprod_logger = ReprodLogger()
reprod_logger.add("map", np.array([0.6847], dtype="float32"))
reprod_logger.save("train_align_benchmark.npy")


if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    benchmark_info = diff_helper.load_info("./train_align_benchmark.npy")
    paddle_info = diff_helper.load_info(
        "yolo_paddle/train_align_paddle.npy")

    diff_helper.compare_info(benchmark_info, paddle_info)

    diff_helper.report(path="train_align_diff.log", diff_threshold=0.0015)
