from lpnrecog.cli import build_parser


def test_ocr_cli_defaults_to_cpu():
    args = build_parser().parse_args(["--input", "image.jpg"])
    assert args.ocr_device == "cpu"
    assert args.ocr_python is None


def test_ocr_cli_accepts_all_device_choices():
    parser = build_parser()
    for device in ("cpu", "cuda", "auto"):
        args = parser.parse_args(
            ["--input", "image.jpg", "--ocr-device", device]
        )
        assert args.ocr_device == device
