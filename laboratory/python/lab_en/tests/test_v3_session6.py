"""Tests for Session 6: CorpusBuilder v3."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from corpus_builder_v3 import (
    CorpusBuilder, CorpusConfig, CurriculumStage, build_curriculum,
)


class TestCorpusConfig:

    def test_default_config(self):
        cfg = CorpusConfig()
        assert cfg.max_bytes == 20_000_000
        assert cfg.dedup_jaccard_threshold == 0.8
        assert "*.py" in cfg.include_patterns


class TestCorpusBuilder:

    def test_build_from_lab_dir(self):
        """Build should produce a non-empty corpus from the lab directory."""
        lab_dir = Path(__file__).resolve().parent.parent
        builder = CorpusBuilder(CorpusConfig(max_bytes=50_000))
        corpus = builder.build(roots=[lab_dir])
        assert len(corpus) > 0
        assert len(corpus) <= 50_000

    def test_build_respects_max_bytes(self):
        """Corpus size should not exceed max_bytes."""
        builder = CorpusBuilder(CorpusConfig(max_bytes=10_000))
        corpus = builder.build(roots=[Path(__file__).resolve().parent.parent])
        assert len(corpus) <= 10_000

    def test_stats_are_populated(self):
        """After build, stats should have non-zero values."""
        builder = CorpusBuilder(CorpusConfig(max_bytes=20_000))
        builder.build(roots=[Path(__file__).resolve().parent.parent])
        s = builder.stats
        assert s["files_scanned"] > 0
        assert s["lines_total"] > 0

    def test_summary_is_string(self):
        builder = CorpusBuilder(CorpusConfig(max_bytes=5_000))
        builder.build(roots=[Path(__file__).resolve().parent.parent])
        assert isinstance(builder.summary(), str)
        assert "Files:" in builder.summary()

    def test_exclude_patterns_work(self):
        """Test files should be excluded by the test_ pattern."""
        builder = CorpusBuilder(CorpusConfig(
            max_bytes=100_000,
            exclude_patterns=(r"test_",),
        ))
        builder.build(roots=[Path(__file__).resolve().parent.parent])
        # Some files should be excluded by the pattern.
        assert builder.stats["files_excluded_pattern"] > 0

    def test_quality_filter_drops_short_lines(self):
        """Lines shorter than min_line_length should be filtered."""
        builder = CorpusBuilder(CorpusConfig(
            max_bytes=10_000, min_line_length=20,
        ))
        builder.build(roots=[Path(__file__).resolve().parent.parent])
        assert builder.stats["lines_filtered"] > 0

    def test_dedup_removes_duplicates(self):
        """Duplicate lines should be detected and removed."""
        builder = CorpusBuilder(CorpusConfig(max_bytes=50_000))
        corpus = builder.build(roots=[Path(__file__).resolve().parent.parent])
        assert builder.stats["lines_dedup"] > 0

    def test_byte_entropy_low_for_repetitive(self):
        """A highly repetitive line should have low entropy."""
        line = b"aaaaaaaaaaaaaaaa"
        ent = CorpusBuilder._byte_entropy(line)
        assert ent < 1.0  # very low entropy

    def test_byte_entropy_high_for_diverse(self):
        """A diverse line should have high entropy."""
        line = b"The quick brown fox jumps over 123 dogs! @#$%^&*()"
        ent = CorpusBuilder._byte_entropy(line)
        assert ent > 3.0  # high entropy

    def test_fingerprint_is_deterministic(self):
        """Same line should produce the same fingerprint."""
        line = b"def hello_world():"
        fp1 = CorpusBuilder._line_fingerprint(line)
        fp2 = CorpusBuilder._line_fingerprint(line)
        assert fp1 == fp2

    def test_fingerprint_normalizes_whitespace(self):
        """Lines differing only in whitespace should have the same fingerprint."""
        line1 = b"def  hello():"
        line2 = b"def hello():"
        fp1 = CorpusBuilder._line_fingerprint(line1)
        fp2 = CorpusBuilder._line_fingerprint(line2)
        assert fp1 == fp2


class TestCurriculum:

    def test_build_curriculum_returns_stages(self):
        """build_curriculum should return n_stages stages."""
        corpus = b"doc1\n=== DOC ===\ndoc2\n=== DOC ===\ndoc3" * 10
        stages = build_curriculum(corpus, n_stages=3, total_epochs=150)
        assert len(stages) == 3
        assert all(isinstance(s, CurriculumStage) for s in stages)

    def test_curriculum_epochs_are_contiguous(self):
        """Stages should cover all epochs without gaps."""
        corpus = b"doc\n=== DOC ===\ndoc" * 5
        stages = build_curriculum(corpus, n_stages=3, total_epochs=150)
        assert stages[0].start_epoch == 0
        assert stages[-1].end_epoch == 150
        for i in range(len(stages) - 1):
            assert stages[i].end_epoch == stages[i + 1].start_epoch

    def test_curriculum_stage_names(self):
        """Each stage should have a name."""
        corpus = b"doc\n=== DOC ===\ndoc" * 5
        stages = build_curriculum(corpus, n_stages=2, total_epochs=100)
        assert all(s.name for s in stages)
