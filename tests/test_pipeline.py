from __future__ import annotations

from pathlib import Path
import unittest

from stt_pipeline.models import CallAnalysis, SalesCallScore, SentimentAnalysis, TranscriptResult
from stt_pipeline.pipeline import _rank_and_label_calls


class BatchRankingTests(unittest.TestCase):
    def test_top_slice_is_shortlisted(self) -> None:
        calls = [
            CallAnalysis(
                transcript=TranscriptResult(source_path=Path(f"candidate-{index}.mp4"), text="sample"),
                score=SalesCallScore(score=score, verdict="shortlist", reasons=[], recommendations=[]),
                sentiment=SentimentAnalysis(compound=0.0, positive=0.0, neutral=1.0, negative=0.0, label="neutral", summary="neutral"),
                transcript_txt_path=Path(f"candidate-{index}.txt"),
                transcript_json_path=Path(f"candidate-{index}.json"),
            )
            for index, score in enumerate([95, 88, 80, 70, 60], start=1)
        ]

        ranked = _rank_and_label_calls(calls, shortlist_ratio=0.2, review_ratio=0.2)

        self.assertEqual([call.rank for call in ranked], [1, 2, 3, 4, 5])
        self.assertEqual(ranked[0].batch_decision, "recommended_shortlist")
        self.assertEqual(ranked[1].batch_decision, "needs_review")
        self.assertEqual(ranked[2].batch_decision, "do_not_shortlist")


if __name__ == "__main__":
    unittest.main()
