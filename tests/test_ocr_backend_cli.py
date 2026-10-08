import unittest

from video_subtitles.cli import build_parser


class OCRBackendCliTests(unittest.TestCase):
    def test_ocr_automatically_selects_backend_and_uses_hardware_default_concurrency(self):
        args = build_parser().parse_args(
            [
                "--input",
                "sample.mp4",
                "--ocr",
            ]
        )

        self.assertEqual(args.ocr_backend, "auto")
        self.assertIsNone(args.ocr_concurrency)
        self.assertIsNone(args.ocr_server_url)

    def test_cli_accepts_llama_cpp_cpu_backend(self):
        args = build_parser().parse_args(
            ["--input", "sample.mp4", "--ocr", "--ocr-backend", "llama-cpp"]
        )

        self.assertEqual(args.ocr_backend, "llama-cpp")

    def test_cli_rejects_removed_local_backend(self):
        with self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(
                ["--input", "sample.mp4", "--ocr", "--ocr-backend", "unknown"]
            )

        self.assertEqual(raised.exception.code, 2)
