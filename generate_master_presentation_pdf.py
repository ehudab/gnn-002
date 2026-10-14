#!/usr/bin/env python3
"""
Pure-stdlib PDF generator for All Tasks Master Presentation and Evaluation Guide.
Compiles the comprehensive, structured Master Defense Guide into an executive multi-page PDF.
Includes:
- Section 0: Foundations of Graph Representation Learning
- Section 1: Task 1 (Fraud Detection from Scratch)
- Section 2: Task 2 (Cora Architecture Comparison)
- Section 3: Task 3 (Link Prediction & Data Leakage Audit)
- Section 4: Task 4 (Production GNN MLOps Pipeline)
- Section 5: Master Examiner Q&A Defense Cheat Sheet
With exact sub-headers: Tracing predictions, Explaining experiments, Defending choices,
Edge cases & failure handling, and Live code modification recipes.
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
    MARGIN_L = 50
    MARGIN_R = 50
    MARGIN_T = 50
    MARGIN_B = 54
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
        if self._cur_y - line_h < self.MARGIN_B + 12:
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
            size = 11.5
            font = 'Helvetica-Bold'
            gap_before = 11
            gap_after = 3
            rgb = (0.10, 0.20, 0.50)
        elif level == 2:
            size = 9.5
            font = 'Helvetica-Bold'
            gap_before = 7
            gap_after = 2
            rgb = (0.15, 0.25, 0.40)
        else:
            size = 8.5
            font = 'Helvetica-Bold'
            gap_before = 5
            gap_after = 2
            rgb = (0.2, 0.2, 0.2)

        line_h = size * 1.25
        self._check_page_break(gap_before + line_h)
        self._cur_y -= gap_before

        if level == 1:
            self._cur_page_lines.append(
                f'{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} rg '
                f'{self.MARGIN_L:.2f} {(self._cur_y - line_h + 2):.2f} '
                f'{self.TEXT_W:.2f} 0.7 re f'.encode()
            )

        self._draw_text(text, font, size, self.MARGIN_L, self._cur_y, rgb)
        self._cur_y -= line_h + gap_after

    def add_paragraph(self, text: str, indent: float = 0, size: float = 8.5,
                      font: str = 'Helvetica', rgb=(0, 0, 0), line_spacing: float = 1.3):
        max_chars = int((self.TEXT_W - indent) / (size * self.CH_WIDTH_RATIO * 0.72))
        max_chars = max(20, min(max_chars, 115))
        wrapped = textwrap.fill(text, width=max_chars)
        line_h = size * line_spacing
        for line in wrapped.split('\n'):
            self._check_page_break(line_h)
            self._draw_text(line, font, size, self.MARGIN_L + indent, self._cur_y, rgb)
            self._cur_y -= line_h

    def add_blank(self, pts: float = 3):
        if self._cur_y - pts < self.MARGIN_B:
            self._end_page()
            self._new_page()
        else:
            self._cur_y -= pts

    def add_rule(self):
        self.add_blank(2)
        self._cur_page_lines.append(
            f'0.75 0.75 0.75 RG 0.5 w '
            f'{self.MARGIN_L:.2f} {self._cur_y:.2f} m '
            f'{self.PAGE_W - self.MARGIN_R:.2f} {self._cur_y:.2f} l S'.encode()
        )
        self.add_blank(3)

    def add_bullet(self, text: str, size: float = 8.5, indent: float = 10):
        self._draw_text('•', 'Helvetica', size, self.MARGIN_L + indent - 7, self._cur_y + size * 0.05)
        self.add_paragraph(text, indent=indent, size=size)

    def add_code_snippet(self, text: str):
        lines = text.strip().split('\n')
        line_h = 10.0
        self._check_page_break(len(lines) * line_h + 5)
        self.add_blank(2)
        for line in lines:
            self._draw_text(line, 'Helvetica-Oblique', 7.5, self.MARGIN_L + 8, self._cur_y, (0.1, 0.3, 0.2))
            self._cur_y -= line_h
        self.add_blank(2)

    def add_page_numbers(self):
        total = len(self._page_streams)
        for i, stream in enumerate(self._page_streams):
            footer = (
                f'BT /Helvetica 7.5 Tf 0.5 0.5 0.5 rg '
                f'{self.PAGE_W / 2 - 25:.2f} {self.MARGIN_B - 18:.2f} Td '
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
        print(f"Master presentation notes saved to: {path}")

def build_pdf(output_path: str):
    pdf = PDF()

    # --- COVER BANNER ---
    pdf._cur_y = PDF.PAGE_H - 80
    pdf._draw_text('Master Presentation & Comprehensive Evaluation Defense Guide', 'Helvetica-Bold', 16, PDF.MARGIN_L, pdf._cur_y, (0.10, 0.20, 0.50))
    pdf._cur_y -= 18
    pdf._draw_text('Foundations, Task 1, Task 2, Task 3, and Task 4 MLOps Pipeline', 'Helvetica-Bold', 11, PDF.MARGIN_L, pdf._cur_y, (0.15, 0.25, 0.45))
    pdf._cur_y -= 14
    pdf._draw_text(f'Author: Naod Abebe · iCog Training · Generated: {datetime.now().strftime("%B %d, %Y")}', 'Helvetica-Oblique', 8.0, PDF.MARGIN_L, pdf._cur_y, (0.55, 0.55, 0.55))
    pdf._cur_y -= 14
    pdf.add_rule()

    # --- SECTION 0: FOUNDATIONS ---
    pdf.add_heading('0. Foundations of Graph Representation Learning', level=1)
    pdf.add_paragraph(
        'Relational data violates the independent and identically distributed (i.i.d.) assumption. Graphs are non-Euclidean, '
        'permutation-invariant, and have variable topology. The Message Passing Neural Network (MPNN) framework updates nodes via '
        'm_u = AGGREGATE({MESSAGE(h_u, h_v) : v in N(u)}) and h_u = COMBINE(h_u, m_u). Spectral GCNs approximate Laplacian graph filters, '
        'while spatial methods (GraphSAGE, GAT) operate directly on local neighborhoods.'
    )
    pdf.add_heading('Presentation and Evaluation (2.5 Minutes):', level=2)
    pdf.add_paragraph(
        'Introduce the graph paradigm: images have rigid translation equivariance; graphs have permutation equivariance: '
        'f(P*X, P*A*P^T) = P*f(X, A). Relational inductive bias allows GNNs to leverage structural homophily to smooth noise.'
    )
    pdf.add_heading('Be Prepared To:', level=2)
    pdf.add_bullet('Trace prediction: On a toy 3-node graph, demonstrate self-loop addition, degree matrix D, normalized adjacency A_hat, and layer aggregation X_agg.')
    pdf.add_bullet('Explain experiments: Contrast graph models with tabular MLPs; show that setting A = I collapses GNN accuracy to the MLP floor.')
    pdf.add_bullet('Defend choices: Spatial message passing over spectral convolutions for scalability and inductive capability.')
    pdf.add_bullet('Edge cases: Isolated nodes handled via self-loops (d_tilde=1); disconnected graphs handled as block-diagonal Laplacians.')
    pdf.add_bullet('Live modification: Implement a 5-line message-passing layer from scratch in pure PyTorch.')

    # --- SECTION 1: TASK 1 ---
    pdf.add_heading('1. Task 1 — Fraud Detection with GCN from Scratch', level=1)
    pdf.add_paragraph('Present: Dataset Design, Implementation, Correctness Checks, Results, and Evidence')
    pdf.add_bullet('Dataset Design: Stochastic Block Model (N=250, 75 Fraud, 175 Legit, 3 features: amount, age, velocity). p_in=0.03, p_out=0.005, edge homophily=0.887.')
    pdf.add_bullet('Implementation: Pure PyTorch without PyG. Degree-normalized adjacency A_hat = D^-0.5 * (A + I) * D^-0.5. 2-layer GCNLayer architecture.')
    pdf.add_bullet('Correctness Checks: Single-node arithmetic match within 1e-6; permutation equivariance error < 1e-7 across 5 random permutations.')
    pdf.add_bullet('Results (5 seeds): GCN Acc=91.2% (F1=0.895, Fraud Recall=88.0%) vs MLP Acc=76.8% (F1=0.720, Fraud Recall=61.3%).')
    pdf.add_bullet('Evidence of Graph Use: Identity adjacency (A=I) collapses GCN to 76.4%; edge corruption sweep drops accuracy monotonically from 91.2% to 72.1%.')
    pdf.add_heading('Presentation and Evaluation (3 Minutes):', level=2)
    pdf.add_paragraph('Walk through the fraud confusion matrix: MLP misses 39% of fraudsters; GCN catches 88% by pooling neighbor cues across shared devices.')
    pdf.add_heading('Be Prepared To:', level=2)
    pdf.add_bullet('Trace prediction: Trace Node 10 with neighbors {12, 15}. Show weights 0.333, 0.289, 0.333, feature blending, ReLU, and softmax.')
    pdf.add_bullet('Explain experiments: Overlapping feature distributions create an unresolvable decision boundary for MLPs that GCN resolves structurally.')
    pdf.add_bullet('Defend choices: Symmetric normalization preserves Laplacian eigenvalues in [0, 2]; class-weighted cross-entropy balances the 70/30 split.')
    pdf.add_bullet('Edge cases: Isolated nodes (degree 0) have d_tilde=1, self-weight 1.0, gracefully executing as an MLP without divide-by-zero crashes.')
    pdf.add_bullet('Live modification: Switch symmetric normalization to row-normalization D^-1 * A; observe hub node dominance.')

    # --- SECTION 2: TASK 2 ---
    pdf.add_heading('2. Task 2 — GNN Architecture Comparison on Cora', level=1)
    pdf.add_paragraph('Present: Comparison, Experiment Design, Error Analysis, and Model Recommendation')
    pdf.add_bullet('Comparison: Benchmarked MLP (59.0%), GCN (81.5%), GraphSAGE (81.0%), and GAT (83.0%) on Cora (2,708 papers, 1,433 features, 7 classes).')
    pdf.add_bullet('Experiment Design: Strictly identical 4-config hyperparameter search on validation data only; test set touched once; 3 seeds reported.')
    pdf.add_bullet('Error Analysis: Misclassifications occur on heterophilous boundary papers (cross-field citations) and low-degree peripheral papers.')
    pdf.add_bullet('Model Recommendation: GAT for citation networks (attention filters noisy citations); GraphSAGE for inductive production; GCN for fast static graphs.')
    pdf.add_heading('Presentation and Evaluation (3 Minutes):', level=2)
    pdf.add_paragraph('Highlight depth sweep: 2 layers achieves 81.5% with cosine similarity 0.24; 4 layers collapses to 58.1% with cosine similarity 0.94 (over-smoothing).')
    pdf.add_heading('Be Prepared To:', level=2)
    pdf.add_bullet('Trace prediction: Trace GAT attention coefficients alpha_uv vs GraphSAGE decoupled self/neighbor linear transformations.')
    pdf.add_bullet('Explain experiments: The 22% gap between MLP and GNNs proves topology dominates; edge-removal sweep shows GNNs degrading toward MLP.')
    pdf.add_bullet('Defend choices: Multi-seed reporting is mandatory because Cora has only 140 training nodes (~5% of graph), causing high seed variance.')
    pdf.add_bullet('Edge cases: Extreme depth over-smoothing; peripheral nodes with degree 1 relying solely on ambiguous bag-of-words features.')
    pdf.add_bullet('Live modification: Add residual skip connections (h = conv(x) + Linear(x)) to GCN to mitigate over-smoothing at depth 3 and 4.')

    # --- SECTION 3: TASK 3 ---
    pdf.add_heading('3. Task 3 — Link Prediction & Data Leakage Audit on Cora', level=1)
    pdf.add_paragraph('Present: Pipeline, Leakage Checks, Decoder Comparison, Failure Cases, and Conclusions')
    pdf.add_bullet('Pipeline: Unique canonical undirected pairs (u < v, 5,278 edges), 70/15/15 split, fixed negative sampling, GraphSAGE encoder.')
    pdf.add_bullet('Leakage Checks: 6 automated assert checks proving val/test positives and reverse copies are absent from message-passing graph.')
    pdf.add_bullet('Decoder Comparison: Dot decoder s(u,v)=sigmoid(z_u · z_v) vs Symmetric MLP s(u,v)=sigmoid(MLP([|z_u-z_v|; z_u*z_v])). Max diff = 0.00e+00.')
    pdf.add_bullet('Failure Cases: False negatives occur on bridge edges across distinct clusters; false positives occur on co-cited papers without direct links.')
    pdf.add_bullet('Conclusions: Clean GNN achieves 0.8979 AUC / 0.9050 AP; Leaky GNN falsely hits 0.9828 AUC (+8.5% inflation) by cheating across test edges.')
    pdf.add_heading('Presentation and Evaluation (3 Minutes):', level=2)
    pdf.add_paragraph('Explain why leaky evaluation is invalid: GNN aggregates features across the test edge, testing memorization rather than generalization.')
    pdf.add_heading('Be Prepared To:', level=2)
    pdf.add_bullet('Trace prediction: Trace pair (548, 809) through GraphSAGE, embedding extraction, and symmetric MLP feature construction.')
    pdf.add_bullet('Explain experiments: Common-neighbor baseline achieves 0.671; GNN achieves 0.898 by fusing topology with 1,433 text features.')
    pdf.add_bullet('Defend choices: Fixed evaluation negatives guarantee reproducible metrics; canonical pairs guarantee both directions reside in the same split.')
    pdf.add_bullet('Edge cases: Proving unobserved links are not confirmed negatives; verifying symmetry assertions on random node pairs.')
    pdf.add_bullet('Live modification: Break symmetry by concatenating [z_u; z_v] in MLPDecoder; run verify_mlp_symmetry() to show immediate AssertionError.')

    # --- SECTION 4: TASK 4 ---
    pdf.add_heading('4. Task 4 — Production GNN MLOps Pipeline', level=1)
    pdf.add_paragraph('Present: Live Demonstration of Training, Tracking, Evaluation, Registration, Promotion Checks, Serving, Monitoring, and Rollback')
    pdf.add_bullet('Live Demo Steps: 1. Untracked baseline flaws; 2. 3 MLflow runs with Git SHA and data hash; 3. PyFunc packaging with reload verification;')
    pdf.add_bullet('4. Model registry gate: auto-approve baseline, promote Champion (Delta=+3.2%), block regression (Delta=-7.2%); 5. Emergency rollback;')
    pdf.add_bullet('6. Containerized FastAPI serving on :8000; 7. Inductive inference on unseen nodes; 8. Structural & embedding drift detection.')
    pdf.add_heading('Presentation and Evaluation (3 Minutes):', level=2)
    pdf.add_paragraph('Emphasize the relational serving contract: GNN APIs must accept neighborhood context rather than isolated feature vectors.')
    pdf.add_heading('Be Prepared To:', level=2)
    pdf.add_bullet('Trace prediction: Trace POST /predict JSON -> boundary validation -> PyFunc wrapper -> GraphSAGE -> softmax -> label mapping.')
    pdf.add_bullet('Explain experiments: High-Capacity Champion (128 hidden, LR 0.005) beat baseline by +3.2%; Max-aggregation failed due to gradient variance.')
    pdf.add_bullet('Defend choices: GraphSAGE over GCN for inductive serving of unseen papers; min_delta=+0.005 threshold to prevent deployment churn.')
    pdf.add_bullet('Edge cases: Handled isolated nodes, duplicate target IDs, invalid edge endpoints 400, bad feature dims 400; latent subspace rotation audit.')
    pdf.add_bullet('Live modification: Increase min_delta to 0.05 in validation_gate.py; demonstrate superior model rejected for insufficient delta.')

    # --- SECTION 5: MASTER Q&A ---
    pdf.add_heading('5. Master Q&A Defense Quick-Reference', level=1)
    pdf.add_paragraph('1. Transductive vs Inductive: Transductive requires full Laplacian of all nodes at train time; Inductive learns local aggregation functions applied anywhere.', font='Helvetica-Bold')
    pdf.add_bullet('2. Why not concatenate [z_u; z_v]? Concatenation is order-dependent, producing two different scores for an undirected link.')
    pdf.add_bullet('3. Why not compare raw embedding coordinates across versions? Neural networks have rotational symmetries (z\' = R*z); raw distances trigger false drift alerts.')
    pdf.add_bullet('4. Why is Average Precision reported? Link graphs have high class imbalance (100:1); AP emphasizes top-ranked precision.')

    pdf.save(output_path)

if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else 'Presentation notes — All Tasks Master Evaluation Guide.pdf'
    build_pdf(out)
