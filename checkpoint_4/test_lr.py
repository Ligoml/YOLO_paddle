from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    torch_info = diff_helper.load_info("yolo_torch/lr_torch.npy")
    paddle_info = diff_helper.load_info("yolo_paddle/lr_paddle.npy")

    diff_helper.compare_info(torch_info, paddle_info)

    diff_helper.report(path="lr_diff.log")
