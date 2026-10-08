import unittest

from video_subtitles.ocr import (
    CPUFrameSampler,
    OCRText,
    PaddleOCRVLEngine,
    _caption_from_entries,
    _has_caption_region_text,
    _paddle_result,
    subtitle_from_frames,
)


class _SequenceEngine:
    engine_name = "test"
    device_name = "test"

    def __init__(self, results):
        self.results = iter(results)

    def recognize_batch(self, images):
        return [next(self.results) for _ in images]

    def close(self):
        pass


class PaddleOCRQualityTests(unittest.TestCase):
    def test_cpu_ocr_input_keeps_gpu_comparable_pixel_budget(self):
        import numpy as np

        engine = PaddleOCRVLEngine.__new__(PaddleOCRVLEngine)
        engine.backend = "llama-cpp"
        engine.batch_size = 1
        observed = {}

        def capture(images):
            observed["shape"] = images[0].shape[:2]
            return [{"res": {"spotting_res": {"rec_texts": [], "rec_polys": []}}}]

        engine._predict = capture

        engine.recognize_batch([np.zeros((1080, 1920, 3), dtype=np.uint8)])

        height, width = observed["shape"]
        self.assertLessEqual(height * width, 131_072)
        self.assertGreaterEqual(height * width, 131_072 * 0.95)

    def test_inference_requests_spotting_mode(self):
        class FakePipeline:
            def predict_iter(self, images, **options):
                self.options = options
                return iter([object() for _ in images])

        for backend, expected in (("vllm", "spotting"), ("llama-cpp", "ocr")):
            with self.subTest(backend=backend):
                engine = PaddleOCRVLEngine.__new__(PaddleOCRVLEngine)
                engine.backend = backend
                engine._engine = FakePipeline()

                engine._predict([object()])

                self.assertEqual(engine._engine.options["prompt_label"], expected)

    def test_cpu_ocr_result_uses_reported_text_block(self):
        result = {
            "res": {
                "parsing_res_list": [
                    {"block_content": "A real subtitle", "block_bbox": [0, 0, 1920, 270]}
                ],
                "spotting_res": {"rec_texts": [], "rec_polys": []},
            }
        }

        self.assertEqual(
            _paddle_result(result, allow_full_crop=True),
            [OCRText("A real subtitle", ((0.0, 0.0), (1920.0, 0.0), (1920.0, 270.0), (0.0, 270.0)), None)],
        )

    def test_cpu_frame_sampler_confirms_caption_then_reduces_repeated_calls(self):
        sampler = CPUFrameSampler()

        selected = [
            sampler.should_process(0.00, has_text=True),
            sampler.should_process(0.25, has_text=True),
            sampler.should_process(0.50, has_text=True),
            sampler.should_process(1.25, has_text=True),
            sampler.should_process(1.50, has_text=True),
            sampler.should_process(1.75, has_text=False),
            sampler.should_process(2.00, has_text=False),
        ]

        self.assertEqual(selected, [True, True, False, True, False, True, False])

    def test_cpu_frame_sampler_keeps_legacy_cadence_when_detector_misses_text(self):
        sampler = CPUFrameSampler()
        selected = [
            sampler.should_process(t, has_text=False)
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

        self.assertEqual(selected, [True, False, True, False, True])

    def test_cpu_frame_sampler_skips_detector_negative_frames(self):
        sampler = CPUFrameSampler()
        selected = [
            sampler.should_process(t, has_text=False, use_fallback=False)
            for t in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]

        self.assertEqual(selected, [False, False, False, False, False])

    def test_cpu_frame_sampler_triggers_when_candidate_layout_changes(self):
        sampler = CPUFrameSampler()
        original_layout = ((2, 18, 12),)
        changed_layout = ((2, 12, 12),)

        selected = [
            sampler.should_process(0.0, True, original_layout),
            sampler.should_process(0.5, True, original_layout),
            sampler.should_process(0.75, True, changed_layout),
            sampler.should_process(1.0, True, changed_layout),
        ]

        self.assertEqual(selected, [True, True, True, False])

    def test_cpu_candidate_detector_requires_caption_like_box_geometry(self):
        self.assertTrue(
            _has_caption_region_text(
                [[[100, 80], [700, 80], [700, 150], [100, 150]]], 1000, 250
            )
        )
        self.assertFalse(
            _has_caption_region_text(
                [[[100, 5], [700, 5], [700, 30], [100, 30]]], 1000, 250
            )
        )
        self.assertFalse(
            _has_caption_region_text(
                [[[100, 10], [120, 10], [120, 240], [100, 240]]], 1000, 250
            )
        )
        self.assertFalse(
            _has_caption_region_text(
                [[[500, 50], [550, 50], [550, 160], [500, 160]]], 1000, 250
            )
        )

    def test_cpu_engine_only_sends_detected_caption_frames_to_vlm(self):
        class _DetectorResult:
            def __init__(self, polygons):
                self.json = {"res": {"dt_polys": polygons}}

        class _Detector:
            def __init__(self):
                self.results = iter(
                    [
                        _DetectorResult([[[100, 80], [1000, 80], [1000, 140], [100, 140]]]),
                        _DetectorResult([[[100, 80], [1000, 80], [1000, 140], [100, 140]]]),
                        _DetectorResult([]),
                    ]
                )

            def predict_iter(self, images):
                return iter([next(self.results)])

        import numpy as np

        engine = PaddleOCRVLEngine.__new__(PaddleOCRVLEngine)
        engine._candidate_detector = _Detector()
        engine._frame_sampler = CPUFrameSampler()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        self.assertTrue(engine.should_process_frame(0.0, frame))
        self.assertTrue(engine.should_process_frame(0.2, frame))
        self.assertFalse(engine.should_process_frame(0.4, frame))

    def test_parses_spotting_polygons_as_text_boxes(self):
        result = {
            "res": {
                "parsing_res_list": [{"block_content": "", "block_bbox": [0, 0, 1920, 270]}],
                "spotting_res": {
                    "rec_texts": ["真实字幕"],
                    "rec_polys": [[[100, 180], [500, 180], [500, 220], [100, 220]]],
                },
            }
        }

        self.assertEqual(
            _paddle_result(result),
            [OCRText("真实字幕", ((100.0, 180.0), (500.0, 180.0), (500.0, 220.0), (100.0, 220.0)), None)],
        )

    def test_does_not_fall_back_to_full_crop_ocr_block_without_spotting_box(self):
        result = {
            "res": {
                "parsing_res_list": [{"block_content": "FF", "block_bbox": [0, 0, 1920, 270]}],
                "spotting_res": {"rec_texts": [], "rec_polys": []},
            }
        }

        self.assertEqual(_paddle_result(result), [])

    def test_short_caption_with_real_box_survives_temporal_consistency_gate(self):
        frame = _Frame()
        short_caption = [
            OCRText("F", ((800, 880), (850, 880), (850, 930), (800, 930)), None)
        ]
        engine = _SequenceEngine([short_caption, short_caption])

        srt = subtitle_from_frames([(0.0, frame), (0.5, frame)], engine)

        self.assertIn("\nF\n", srt)

    def test_single_frame_hallucination_is_not_emitted(self):
        frame = _Frame()
        false_positive = [
            OCRText("FF", ((800, 880), (850, 880), (850, 930), (800, 930)), None)
        ]
        engine = _SequenceEngine([false_positive, []])

        self.assertIsNone(subtitle_from_frames([(0.0, frame), (0.5, frame)], engine))

    def test_caption_box_must_be_in_subtitle_region(self):
        frame = _Frame()
        upper_frame_text = OCRText(
            "频道名", ((100, 100), (300, 100), (300, 140), (100, 140)), None
        )

        self.assertIsNone(_caption_from_entries([upper_frame_text], frame))


class _Frame:
    shape = (1080, 1920, 3)


if __name__ == "__main__":
    unittest.main()
