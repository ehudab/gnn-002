#!/usr/bin/env python3
"""
Pure-stdlib PDF generator for Task 4 Presentation Notes.
Produces an executive presentation document matching the style of Task 1, 2, and 3.
"""

from __future__ import annotations
import struct
import zlib
import textwrap
import io
import os
import sys
from datetime import datetime

class PDF:
    PAGE_W = 595.28
    PAGE_H = 841.89
    MARGIN_L = 60
    MARGIN_R = 60
    MARGIN_T = 60
    MARGIN_B = 65
    TEXT_W = PAGE_W - MARGIN_L - MARGIN_R
    CH_WIDTH_RATIO = 0.556

    def __init__(self):
        self._objects: list[bytes] = []
        self._offsets: list[int] = []
        self._page_streams: list[bytes] = []
        self._cur_page_lines: list[bytes] = []
        self._cur_y: float = self.PAGE_H - self.MARGIN_T
        self._cur_page_num: int = 0
        self._new_page()

    def _escape(self, text: str) -> bytes:
        return (
            text.replace('\\', '\\\\')
                .replace('(', '\\(')
                .replace(')', '\\)')
        ).encode('latin-1', errors='replace')

    def _new_page(self):
        self._cur_y = self.PAGE_H - self.MARGIN_T
        self._cur_page_num += 1
        self._cur_page_lines = []
        self._page_streams.append(b'')

    def _end_page(self):
        self._page_streams[-1] = b'\n'.join(self._cur_page_lines)

    def _check_page_break(self, line_h: float):
        if self._cur_y - line_h < self.MARGIN_B + 15:
            self._end_page()
            self._new_page()

    def _draw_text(self, text: str, font: str, size: float, x: float, y: float, rgb=(0, 0, 0)):
        r, g, b = rgb
        self._cur_page_lines.append(
            f'BT /{font} {size:.1f} Tf {r:.2f} {g:.2f} {b:.2f} rg '
            f'{x:.2f} {y:.2f} Td ({self._escape(text).decode("latin-1")}) Tj ET'.encode()
        )

    def add_heading(self, text: str, level: int = 1):
        if level == 1:
            size = 13.0
            font = 'Helvetica-Bold'
            gap_before = 14
            gap_after = 4
            rgb = (0.10, 0.20, 0.50)
        else:
            size = 10.5
            font = 'Helvetica-Bold'
            gap_before = 9
            gap_after = 3
            rgb = (0.15, 0.15, 0.15)

        line_h = size * 1.35
        self._check_page_break(gap_before + line_h)
        self._cur_y -= gap_before

        if level == 1:
            self._cur_page_lines.append(
                f'{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} rg '
                f'{self.MARGIN_L:.2f} {(self._cur_y - line_h + 2):.2f} '
                f'{self.TEXT_W:.2f} 0.8 re f'.encode()
            )

        self._draw_text(text, font, size, self.MARGIN_L, self._cur_y, rgb)
        self._cur_y -= line_h + gap_after

    def add_paragraph(self, text: str, indent: float = 0, size: float = 9.5,
                      font: str = 'Helvetica', rgb=(0, 0, 0), line_spacing: float = 1.4):
        max_chars = int((self.TEXT_W - indent) / (size * self.CH_WIDTH_RATIO * 0.72))
        max_chars = max(20, min(max_chars, 105))
        wrapped = textwrap.fill(text, width=max_chars)
        line_h = size * line_spacing
        for line in wrapped.split('\n'):
            self._check_page_break(line_h)
            self._draw_text(line, font, size, self.MARGIN_L + indent, self._cur_y, rgb)
            self._cur_y -= line_h

    def add_blank(self, pts: float = 5):
        if self._cur_y - pts < self.MARGIN_B:
            self._end_page()
            self._new_page()
        else:
            self._cur_y -= pts

    def add_rule(self):
        self.add_blank(4)
        self._cur_page_lines.append(
            f'0.75 0.75 0.75 RG 0.5 w '
            f'{self.MARGIN_L:.2f} {self._cur_y:.2f} m '
            f'{self.PAGE_W - self.MARGIN_R:.2f} {self._cur_y:.2f} l S'.encode()
        )
        self.add_blank(5)

    def add_bullet(self, text: str, size: float = 9.5, indent: float = 14):
        self._draw_text('•', 'Helvetica', size, self.MARGIN_L + indent - 10, self._cur_y + size * 0.05)
        self.add_paragraph(text, indent=indent, size=size)

    def add_code_snippet(self, text: str):
        lines = text.strip().split('\n')
        line_h = 11.0
        self._check_page_break(len(lines) * line_h + 8)
        self.add_blank(3)
        for line in lines:
            self._draw_text(line, 'Helvetica-Oblique', 8.5, self.MARGIN_L + 12, self._cur_y, (0.1, 0.3, 0.2))
            self._cur_y -= line_h
        self.add_blank(3)

    def add_page_numbers(self):
        total = len(self._page_streams)
        for i, stream in enumerate(self._page_streams):
            footer = (
                f'BT /Helvetica 8 Tf 0.5 0.5 0.5 rg '
                f'{self.PAGE_W / 2 - 25:.2f} {self.MARGIN_B - 22:.2f} Td '
                f'(Page {i+1} of {total}) Tj ET'
            ).encode()
            self._page_streams[i] = stream + b'\n' + footer

    def save(self, path: str):
        self._end_page()
        self.add_page_numbers()

        buf = io.BytesIO()
        buf.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
        offsets = []
        obj_num = 0

        def next_obj(body: bytes) -> int:
            nonlocal obj_num
            obj_num += 1
            offsets.append(buf.tell())
            buf.write(f'{obj_num} 0 obj\n'.encode())
            buf.write(body)
            buf.write(b'\nendobj\n')
            return obj_num

        def font_obj(base_name: str) -> int:
            return next_obj(
                f'<< /Type /Font /Subtype /Type1 /BaseFont /{base_name} '
                f'/Encoding /WinAnsiEncoding >>'.encode()
            )

        f_helv = font_obj('Helvetica')
        f_helv_b = font_obj('Helvetica-Bold')
        f_helv_o = font_obj('Helvetica-Oblique')

        font_res = (
            f'/Font << /Helvetica {f_helv} 0 R '
            f'/Helvetica-Bold {f_helv_b} 0 R '
            f'/Helvetica-Oblique {f_helv_o} 0 R >>'
        )

        page_obj_nums = []
        for stream in self._page_streams:
            compressed = zlib.compress(stream)
            cs_num = next_obj(
                f'<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n'.encode()
                + compressed + b'\nendstream'
            )
            pg_num = next_obj(
                f'<< /Type /Page /MediaBox [0 0 {self.PAGE_W:.2f} {self.PAGE_H:.2f}] '
                f'/Contents {cs_num} 0 R '
                f'/Resources << {font_res} >> '
                f'/Parent PAGES >>'.encode()
            )
            page_obj_nums.append(pg_num)

        kids = ' '.join(f'{n} 0 R' for n in page_obj_nums)
        pages_num = next_obj(
            f'<< /Type /Pages /Count {len(page_obj_nums)} /Kids [{kids}] >>'.encode()
        )

        catalog_num = next_obj(
            f'<< /Type /Catalog /Pages {pages_num} 0 R >>'.encode()
        )

        # Patch pages
        for pg_num in page_obj_nums:
            offsets[pg_num - 1] = buf.tell()
            buf.write(f'{pg_num} 0 obj\n'.encode())
            buf.write(
                f'<< /Type /Page /MediaBox [0 0 {self.PAGE_W:.2f} {self.PAGE_H:.2f}] '
                f'/Contents {pg_num - 1} 0 R '
                f'/Resources << {font_res} >> '
                f'/Parent {pages_num} 0 R >>'.encode()
            )
            buf.write(b'\nendobj\n')

        xref_pos = buf.tell()
        buf.write(f'xref\n0 {obj_num + 1}\n'.encode())
        buf.write(b'0000000000 65535 f \n')
        for off in offsets:
            buf.write(f'{off:010d} 00000 n \n'.encode())

        buf.write(
            f'trailer\n<< /Size {obj_num + 1} /Root {catalog_num} 0 R >>\n'
            f'startxref\n{xref_pos}\n%%EOF\n'.encode()
        )

        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        print(f"Presentation notes saved to: {path}")

def build_pdf(output_path: str):
    pdf = PDF()

    # --- TITLE BANNER ---
    pdf._cur_y = PDF.PAGE_H - 120
    pdf._draw_text('Presentation Notes — Task 4', 'Helvetica-Bold', 20, PDF.MARGIN_L, pdf._cur_y, (0.10, 0.20, 0.50))
    pdf._cur_y -= 26
    pdf._draw_text('Production MLOps Pipeline for Graph Neural Networks', 'Helvetica-Bold', 15, PDF.MARGIN_L, pdf._cur_y, (0.10, 0.20, 0.50))
    pdf._cur_y -= 20
    pdf._draw_text('Live Demonstration Guide: Training, Tracking, Packaging, Serving, Gate & Rollback', 'Helvetica', 11, PDF.MARGIN_L, pdf._cur_y, (0.35, 0.35, 0.35))
    pdf._cur_y -= 16
    pdf._draw_text(f'Task 4 · iCog GNN Training · Generated: {datetime.now().strftime("%B %d, %Y")}', 'Helvetica-Oblique', 9, PDF.MARGIN_L, pdf._cur_y, (0.55, 0.55, 0.55))
    pdf._cur_y -= 25
    pdf.add_rule()

    # --- SECTION 1 ---
    pdf.add_heading('1. Executive Demonstration Plan & Narrative Arc', level=1)
    pdf.add_paragraph(
        'The purpose of this live demonstration is to show how an academic Graph Neural Network '
        '(GraphSAGE node classifier on Cora) is transformed into an enterprise-grade production MLOps system. '
        'Traditional ML serving paradigms assume independent and identically distributed (i.i.d.) samples; '
        'GNNs, however, introduce relational graph-context dependencies, topological failure modes, and '
        'latent subspace rotation during drift monitoring. This presentation walks through 8 structured scenes '
        'from bare untracked training to automated rollback.'
    )
    pdf.add_blank(4)
    pdf.add_bullet('Scene 1: Expose anti-patterns in bare untracked training (train.py)')
    pdf.add_bullet('Scene 2: Run 3 reproducible tracked experiments with MLflow (train_tracked.py)')
    pdf.add_bullet('Scene 3: Package model with PyFunc, schema definitions, and reload verification')
    pdf.add_bullet('Scene 4: Execute model registry validation gate: promote champion & block regressions')
    pdf.add_bullet('Scene 5: Deploy containerized FastAPI & MLflow services via Docker Compose')
    pdf.add_bullet('Scene 6: Test inductive serving on unseen nodes & strict boundary failure modes')
    pdf.add_bullet('Scene 7: Detect structural and embedding representation drift with fixed reference models')
    pdf.add_bullet('Scene 8: Demonstrate automated emergency rollback restoring the previous stable version')

    # --- SECTION 2 ---
    pdf.add_heading('2. Scene 1 — The Untracked Baseline & Production Anti-Patterns', level=1)
    pdf.add_paragraph('Say out loud:', font='Helvetica-Bold', rgb=(0.1, 0.3, 0.1))
    pdf.add_paragraph(
        '"In production, model weights alone are completely useless without provenance, metadata, and schemas. '
        'Here is how typical research scripts train GNNs—and why they fail immediately in real-world deployments."',
        indent=12, font='Helvetica-Oblique'
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Command to run in terminal:')
    pdf.add_code_snippet('python task_4/train.py')
    pdf.add_paragraph('Five Critical Flaws Highlighted to the Audience:')
    pdf.add_bullet('No Lineage or Tracking: Hyperparameters and training loss curves vanish when the terminal closes.')
    pdf.add_bullet('Zero Reproducibility: No record of Git commit SHA, dataset snapshot hash, or random seeds.')
    pdf.add_bullet('Brittle Artifact Packaging: Saving a bare state_dict (.pt file) requires copy-pasting the exact Python class definition in the serving server.')
    pdf.add_bullet('Blind Overwrites: The script blindly overwrites baseline_model.pt without comparing validation accuracy against the currently running champion.')
    pdf.add_bullet('No Serving Contract: No schema exists specifying that Cora requires 1,433-dim features and maps outputs to 7 classes.')

    # --- SECTION 3 ---
    pdf.add_heading('3. Scene 2 & 3 — Experiment Tracking, Lineage & Model Packaging', level=1)
    pdf.add_paragraph('Say out loud:', font='Helvetica-Bold', rgb=(0.1, 0.3, 0.1))
    pdf.add_paragraph(
        '"Our tracked pipeline records everything in MLflow: code commit SHA, dataset SHA-256 hash, '
        'and 3 controlled experiments. Furthermore, we wrap the model in a custom MLflow PyFunc container '
        'that bundles weights, hyperparameters, and feature schemas."',
        indent=12, font='Helvetica-Oblique'
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Command to run in terminal:')
    pdf.add_code_snippet('python task_4/train_tracked.py --run-all')
    pdf.add_paragraph('The Three Benchmark Experiments:')
    pdf.add_bullet('Exp 1 (Baseline Candidate): Seed 42, Hidden Dim 64, LR 0.01, Mean Aggregation (Val Acc: ~81.2%)')
    pdf.add_bullet('Exp 2 (High Capacity Champion): Seed 101, Hidden Dim 128, LR 0.005, Mean Aggregation (Val Acc: ~84.4%)')
    pdf.add_bullet('Exp 3 (Regression Foil): Seed 2024, Hidden Dim 32, LR 0.03, Max Aggregation (Val Acc: ~77.2%)')
    pdf.add_blank(4)
    pdf.add_paragraph('Reloading Reproducibility Check:')
    pdf.add_paragraph(
        'During artifact logging, train_tracked.py instantly reloads the packaged PyFunc model and passes '
        'sample test nodes to verify: reloaded_preds == training_runtime_preds (100% agreement guaranteed).'
    )

    # --- SECTION 4 ---
    pdf.add_heading('4. Scene 4 — Model Registry, Promotion Gate & Blocked Regression', level=1)
    pdf.add_paragraph('Say out loud:', font='Helvetica-Bold', rgb=(0.1, 0.3, 0.1))
    pdf.add_paragraph(
        '"Engineers should never manually deploy models. We enforce automated validation gates '
        'governed by strict mathematical criteria: first-model floor, minimum delta improvement, and conservative tie-breaking."',
        indent=12, font='Helvetica-Oblique'
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Demonstration Commands:')
    pdf.add_code_snippet(
        '# Step 1: Auto-approve initial model meeting floor (val_acc >= 0.70)\n'
        'python task_4/validation_gate.py --candidate task_4/artifacts/exp_1_graphsage_baseline/run_summary.json\n\n'
        '# Step 2: Promote superior model (delta >= +0.005)\n'
        'python task_4/validation_gate.py --candidate task_4/artifacts/exp_2_graphsage_highcap/run_summary.json\n\n'
        '# Step 3: Block regression candidate\n'
        'python task_4/validation_gate.py --candidate task_4/artifacts/exp_3_graphsage_max_aggr/run_summary.json'
    )
    pdf.add_paragraph('Governance Rules Explained to Audience:')
    pdf.add_bullet('First-Model Rule: Auto-promotes to Version 1 if val_acc exceeds the 70% quality floor.')
    pdf.add_bullet('Threshold Rule (min_delta = +0.005): Must exceed current champion by >= 0.5% on identical validation data.')
    pdf.add_bullet('Tie Behavior: If delta is between 0.0 and +0.005, candidate is REJECTED to avoid deployment churn.')
    pdf.add_bullet('Blocked Regression: Exp 3 (77.2% vs 84.4%) is rejected with explicit audit reason.')
    pdf.add_bullet('Fail-Closed Rule: Missing or corrupted metrics trigger immediate rejection.')

    # --- SECTION 5 ---
    pdf.add_heading('5. Scene 5 & 6 — Containerized Serving, Inductive Inference & Failure Modes', level=1)
    pdf.add_paragraph('Say out loud:', font='Helvetica-Bold', rgb=(0.1, 0.3, 0.1))
    pdf.add_paragraph(
        '"Our FastAPI service is containerized via Docker Compose. It serves both transductive and inductive '
        'graph queries. Because GraphSAGE learns localized message-passing aggregations rather than static embeddings, '
        'we can classify brand-new, unseen nodes added after training."',
        indent=12, font='Helvetica-Oblique'
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Demonstration Commands:')
    pdf.add_code_snippet(
        '# Health & loaded model metadata\n'
        'curl -s http://localhost:8000/health\n'
        'curl -s http://localhost:8000/model-info\n\n'
        '# Execute full integration & failure test suite\n'
        'python -m unittest task_4/tests/test_api.py'
    )
    pdf.add_paragraph('Six Robustness Cases Demonstrated Live:')
    pdf.add_bullet('Inductive Serving: Unseen test papers excluded from training are classified with valid confidences.')
    pdf.add_bullet('Offline vs API Consistency: 100% agreement between direct PyTorch inference and FastAPI JSON response.')
    pdf.add_bullet('Isolated Nodes (Degree 0): Handled gracefully via self-transformation without crashing.')
    pdf.add_bullet('Duplicate Target IDs ([1, 1, 0]): Automatically deduplicated into [1, 0] preserving order.')
    pdf.add_bullet('Invalid Edge Endpoints (0 -> 99 on 3-node graph): Returns HTTP 400 Bad Request.')
    pdf.add_bullet('Incorrect Feature Dimension (100 dims instead of 1433): Returns HTTP 400 Bad Request.')

    # --- SECTION 6 ---
    pdf.add_heading('6. Scene 7 — Drift Monitoring & The Subspace Rotation Trap', level=1)
    pdf.add_paragraph('Say out loud:', font='Helvetica-Bold', rgb=(0.1, 0.3, 0.1))
    pdf.add_paragraph(
        '"Graph data drifts in two ways: structural topology and representation embeddings. '
        'However, comparing raw embedding coordinates between different model versions is a dangerous trap '
        'due to latent coordinate rotation."',
        indent=12, font='Helvetica-Oblique'
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Why Raw Embedding Coordinates Cannot Be Compared Across Models:')
    pdf.add_bullet('Non-Convex Latent Symmetries: Retraining neural networks with different random seeds or hyperparameter sets induces arbitrary orthogonal rotations (z_new = R * z, R^T * R = I).')
    pdf.add_bullet('False Drift Alerts: Two models with identical 85% accuracy will show massive Euclidean distance shifts between their raw coordinates.')
    pdf.add_bullet('Our Production Solution: We monitor snapshots using a fixed reference model, and track invariant metrics (mean L2 vector norm and degree distribution shifts).')
    pdf.add_blank(4)
    pdf.add_paragraph('Demonstration Command:')
    pdf.add_code_snippet(
        '# Structural (20% tolerance) and Embedding Norm (15% tolerance) Drift Checks\n'
        'python -c "from task_4.drift.monitor import *; print(\'Drift Monitor Ready\')"'
    )

    # --- SECTION 7 ---
    pdf.add_heading('7. Scene 8 — CI Automation & Instant Emergency Rollback', level=1)
    pdf.add_paragraph('Say out loud:', font='Helvetica-Bold', rgb=(0.1, 0.3, 0.1))
    pdf.add_paragraph(
        '"In our GitHub Actions CI pipeline, failing a unit test or regression check blocks deployment. '
        'If an unexpected bug occurs in production, we execute an instantaneous rollback restoring the previous stable champion."',
        indent=12, font='Helvetica-Oblique'
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Live Emergency Rollback Command:')
    pdf.add_code_snippet('python task_4/validation_gate.py --rollback')
    pdf.add_paragraph('Expected Screen Output:')
    pdf.add_code_snippet(
        'ROLLBACK SUCCESSFUL: Restored \'exp_1_graphsage_baseline\' as Version 3.'
    )
    pdf.add_paragraph('What Happens Internally:')
    pdf.add_bullet('Registry State Update: The regressed champion is marked ROLLED_BACK in audit history.')
    pdf.add_bullet('Version Bump: The previous champion is reactivated under a monotonically increasing version number (Version 3).')
    pdf.add_bullet('Hot-Reloading: Calling POST /reload-model on FastAPI instantly switches the in-memory PyFunc model with zero container downtime.')

    # --- SECTION 8 ---
    pdf.add_heading('8. Summary Table & Key Presentation Takeaways', level=1)
    pdf.add_paragraph(
        '1. GNNs Require Relational MLOps: Serving must preserve neighborhood context rather than treating nodes as isolated vectors.\n'
        '2. Pinned Provenance: Code SHA, dataset checksums, and random seeds guarantee 100% auditability.\n'
        '3. Fail-Closed Validation Gates: Mathematical promotion rules protect production from performance regressions.\n'
        '4. Containerized Decoupling: FastAPI + MLflow in Docker Compose provides local reproducibility and cloud portability.\n'
        '5. Safe Drift Auditing: Invariant metrics and fixed reference models prevent false alarms caused by latent subspace rotation.'
    )

    pdf.save(output_path)

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'Presentation notes — Production GNN MLOps Pipeline.pdf'
    build_pdf(out)

