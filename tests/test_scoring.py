from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stt_pipeline.models import TranscriptResult
from stt_pipeline.scoring import CandidateShortlistScorer


class CandidateShortlistScorerTests(unittest.TestCase):
    def test_scores_relevant_experience_higher(self) -> None:
        scorer = CandidateShortlistScorer()
        transcript = TranscriptResult(
            source_path=Path("sample.mp4"),
            text=(
                "I have worked on a Python and SQL project. I learned machine learning in college. "
                "I am eager to grow and contribute to the team."
            ),
        )

        score = scorer.score(transcript)

        self.assertGreaterEqual(score.score, 60)
        self.assertEqual(score.verdict, "hold")

    def test_scores_low_when_candidate_lacks_signal(self) -> None:
        scorer = CandidateShortlistScorer()
        transcript = TranscriptResult(
            source_path=Path("sample.mp4"),
            text="I don't have much experience and I am not really familiar with the tools.",
        )

        score = scorer.score(transcript)

        self.assertLess(score.score, 55)
        self.assertEqual(score.verdict, "no_shortlist")


if __name__ == "__main__":
    unittest.main()
