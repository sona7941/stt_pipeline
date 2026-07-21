from __future__ import annotations

import re
from dataclasses import dataclass

from .models import SalesCallScore, TranscriptResult


_SKILL_PHRASES = (
    "python",
    "sql",
    "machine learning",
    "data analysis",
    "communication",
    "teamwork",
    "problem solving",
    "programming",
    "technical",
)

_PROJECT_PHRASES = (
    "project",
    "built",
    "developed",
    "worked on",
    "assignment",
    "internship",
    "research",
    "capstone",
    "prototype",
)

_MOTIVATION_PHRASES = (
    "looking for",
    "interested",
    "eager",
    "learn",
    "grow",
    "contribute",
    "opportunity",
    "apply",
)

_NEGATIVE_PHRASES = (
    "don't have",
    "not really",
    "not much",
    "not familiar",
    "no experience",
    "little",
    "currently",
    "lack",
)

_STRONG_SIGNAL_PHRASES = (
    "i have",
    "i worked",
    "i built",
    "i learned",
    "i can",
    "i am",
)


@dataclass(frozen=True)
class ScoringWeights:
    discovery: int = 20
    next_step: int = 20
    value: int = 15
    objection_handling: int = 10
    question_density: int = 10
    filler_penalty: int = -10
    close_penalty: int = -15


class CandidateShortlistScorer:
    def __init__(self, weights: ScoringWeights | None = None) -> None:
        self.weights = weights or ScoringWeights()

    def score(self, transcript: TranscriptResult) -> SalesCallScore:
        text = transcript.text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        word_count = max(len(tokens), 1)

        reasons: list[str] = []
        recommendations: list[str] = []
        score = 0

        skill_hits = self._count_phrase_hits(text, _SKILL_PHRASES)
        project_hits = self._count_phrase_hits(text, _PROJECT_PHRASES)
        motivation_hits = self._count_phrase_hits(text, _MOTIVATION_PHRASES)
        negative_hits = self._count_phrase_hits(text, _NEGATIVE_PHRASES)
        strong_signal_hits = self._count_phrase_hits(text, _STRONG_SIGNAL_PHRASES)
        question_hits = transcript.text.count("?")

        skill_score = min(25, skill_hits * 7)
        project_score = min(25, project_hits * 8)
        motivation_score = min(15, motivation_hits * 5)
        confidence_score = min(15, strong_signal_hits * 5)
        interaction_score = 10 if question_hits >= 4 or self._looks_question_heavy(text) else 0
        clarity_score = 10 if word_count >= 80 else 5 if word_count >= 50 else 0

        if skill_hits >= 3:
            reasons.append("The candidate references multiple relevant skills or tools.")
        elif skill_hits >= 1:
            reasons.append("The candidate mentions at least one relevant skill.")
        else:
            recommendations.append("Probe for concrete skills that match the role more closely.")

        if project_hits >= 2:
            reasons.append("The candidate provides concrete project or experience signals.")
        elif project_hits == 1:
            reasons.append("The candidate gives some experience detail, but it is limited.")
        else:
            recommendations.append("Ask for a specific project or example of recent work.")

        if motivation_hits >= 2:
            reasons.append("The candidate expresses clear motivation and growth mindset.")
        elif motivation_hits == 1:
            reasons.append("The candidate expresses some motivation for the role.")

        if strong_signal_hits >= 2:
            reasons.append("The candidate uses confident first-person statements about experience or capability.")
        elif strong_signal_hits == 1:
            reasons.append("The candidate makes at least one confident capability statement.")

        if negative_hits >= 2:
            score -= 15
            recommendations.append("The candidate shows gaps that should be probed before shortlisting.")
        elif negative_hits == 1:
            score -= 8

        if question_hits >= 4 or self._looks_question_heavy(text):
            reasons.append("The interview contains active back-and-forth, which is usually helpful for evaluation.")

        if word_count < 40:
            score -= 5
            recommendations.append("Capture longer answers to better judge the candidate's fit.")

        score += skill_score + project_score + motivation_score + confidence_score + interaction_score + clarity_score
        score = max(0, min(100, score))
        verdict = self._verdict(score)

        if score >= 75:
            recommendations.append("This candidate looks suitable to shortlist for the next round.")
        elif score >= 55:
            recommendations.append("This candidate is borderline and should be held for additional review.")
        else:
            recommendations.append("This candidate should not be shortlisted yet based on the transcript alone.")

        return SalesCallScore(score=score, verdict=verdict, reasons=reasons, recommendations=self._dedupe(recommendations))

    def _count_phrase_hits(self, text: str, phrases: tuple[str, ...]) -> int:
        return sum(1 for phrase in phrases if phrase in text)

    def _looks_question_heavy(self, text: str) -> bool:
        question_starters = (
            "what",
            "how",
            "why",
            "when",
            "where",
            "who",
            "could you",
            "would you",
            "can you",
        )
        lines = re.split(r"[.!?]\s+", text)
        count = sum(1 for line in lines if line.strip().startswith(question_starters))
        return count >= 3

    def _verdict(self, score: int) -> str:
        if score >= 75:
            return "shortlist"
        if score >= 55:
            return "hold"
        return "no_shortlist"

    def _dedupe(self, items: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            if item not in seen:
                unique.append(item)
                seen.add(item)
        return unique


SalesCallQualityScorer = CandidateShortlistScorer
