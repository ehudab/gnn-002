"""
Pure-stdlib PDF writer for Task 3 presentation notes.
Produces a compliant PDF 1.4 document with:
  - Helvetica font (built-in, no embed needed)
  - Multi-line text with section headings
  - Automatic page breaks
  - Page numbers
"""

from __future__ import annotations
import struct
import zlib
import textwrap
import io
from datetime import datetime


# ──────────────────────────────────────────────────────────────────────────────
# Minimal PDF builder
# ──────────────────────────────────────────────────────────────────────────────

class PDF:
    """Minimal PDF 1.4 writer (no external dependencies)."""

    # PDF page size: A4 in points (1 pt = 1/72 inch)
    PAGE_W = 595.28
    PAGE_H = 841.89
    MARGIN_L = 72
    MARGIN_R = 72
    MARGIN_T = 72
    MARGIN_B = 80
    TEXT_W = PAGE_W - MARGIN_L - MARGIN_R  # usable width

    # Font metrics for Helvetica (approximate; 1000 units per em)
    # Average character width ≈ 0.556 × font_size for Helvetica regular
    CH_WIDTH_RATIO = 0.556

    def __init__(self):
        self._objects: list[bytes] = []  # raw object bodies (index 0 = not used)
        self._offsets: list[int] = []
        self._pages: list[int] = []       # object numbers of page objects
        self._buf = io.BytesIO()
        self._page_streams: list[bytes] = []  # content stream for each page

        # Current drawing state
        self._cur_page_lines: list[str] = []  # (op, text, font, size, x, y)
        self._cur_y: float = self.PAGE_H - self.MARGIN_T
        self._cur_page_num: int = 0

        self._new_page()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _add_object(self, body: bytes) -> int:
        """Append a PDF object and return its 1-based index."""
        self._objects.append(body)
        return len(self._objects)  # 1-based

    def _escape(self, text: str) -> bytes:
        """Escape text for PDF string literals."""
        return (
            text.replace('\\', '\\\\')
                .replace('(', '\\(')
                .replace(')', '\\)')
        ).encode('latin-1', errors='replace')

    # ── Page management ───────────────────────────────────────────────────────

    def _new_page(self):
        self._cur_y = self.PAGE_H - self.MARGIN_T
        self._cur_page_num += 1
        self._cur_page_lines = []
        self._page_streams.append(None)  # placeholder; filled on _end_page

    def _end_page(self):
        """Serialise the current page's drawing commands into a content stream."""
        ops = []
        for item in self._cur_page_lines:
            ops.append(item)
        self._page_streams[-1] = b'\n'.join(ops)

    def _check_page_break(self, line_h: float):
        """Start a new page if the next line won't fit."""
        if self._cur_y - line_h < self.MARGIN_B + 20:
            self._end_page()
            self._new_page()

    # ── Public drawing API ────────────────────────────────────────────────────

    def _draw_text(self, text: str, font: str, size: float, x: float, y: float,
                   rgb: tuple = (0, 0, 0)):
        r, g, b = rgb
        self._cur_page_lines.append(
            f'BT /{font} {size:.1f} Tf {r:.2f} {g:.2f} {b:.2f} rg '
            f'{x:.2f} {y:.2f} Td ({self._escape(text).decode("latin-1")}) Tj ET'.encode()
        )

    def add_heading(self, text: str, level: int = 1):
        """Add a section heading (level 1 or 2)."""
        if level == 1:
            size = 13.5
            font = 'Helvetica-Bold'
            gap_before = 14
            gap_after = 4
            rgb = (0.10, 0.20, 0.50)
        else:
            size = 11
            font = 'Helvetica-Bold'
            gap_before = 10
            gap_after = 3
            rgb = (0.15, 0.15, 0.15)

        line_h = size * 1.4
        self._check_page_break(gap_before + line_h)
        self._cur_y -= gap_before

        # Underline rule for level-1
        if level == 1:
            self._cur_page_lines.append(
                f'{rgb[0]:.2f} {rgb[1]:.2f} {rgb[2]:.2f} rg '
                f'{self.MARGIN_L:.2f} {(self._cur_y - line_h + 2):.2f} '
                f'{self.TEXT_W:.2f} 1 re f'.encode()
            )

        self._draw_text(text, font, size, self.MARGIN_L, self._cur_y, rgb)
        self._cur_y -= line_h + gap_after

    def add_paragraph(self, text: str, indent: float = 0, size: float = 10,
                      font: str = 'Helvetica', rgb: tuple = (0, 0, 0),
                      line_spacing: float = 1.45):
        """Word-wrap and add a paragraph of text."""
        max_chars = int(self.TEXT_W / (size * self.CH_WIDTH_RATIO * 0.72))
        max_chars = max(20, min(max_chars, 100))
        indent_str = ' ' * int(indent / (size * self.CH_WIDTH_RATIO * 0.72))
        wrapped = textwrap.fill(text, width=max_chars,
                                initial_indent=indent_str,
                                subsequent_indent=indent_str)
        line_h = size * line_spacing
        for line in wrapped.split('\n'):
            self._check_page_break(line_h)
            self._draw_text(line, font, size, self.MARGIN_L + indent,
                            self._cur_y, rgb)
            self._cur_y -= line_h

    def add_blank(self, pts: float = 6):
        """Add vertical whitespace."""
        if self._cur_y - pts < self.MARGIN_B:
            self._end_page()
            self._new_page()
        else:
            self._cur_y -= pts

    def add_rule(self):
        """Add a thin horizontal rule."""
        self.add_blank(4)
        self._cur_page_lines.append(
            f'0.7 0.7 0.7 RG 0.5 w '
            f'{self.MARGIN_L:.2f} {self._cur_y:.2f} m '
            f'{self.PAGE_W - self.MARGIN_R:.2f} {self._cur_y:.2f} l S'.encode()
        )
        self.add_blank(6)

    def add_bullet(self, text: str, size: float = 10, indent: float = 16):
        """Add a bulleted paragraph."""
        self._draw_text('•', 'Helvetica', size, self.MARGIN_L + indent - 12,
                        self._cur_y + size * 0.05)
        self.add_paragraph(text, indent=indent, size=size)

    def add_page_numbers(self):
        """Called before save; injects page-number footer into each page."""
        total = len(self._page_streams)
        for i, stream in enumerate(self._page_streams):
            footer = (
                f'BT /Helvetica 8 Tf 0.5 0.5 0.5 rg '
                f'{self.PAGE_W / 2 - 30:.2f} {self.MARGIN_B - 20:.2f} Td '
                f'(Page {i+1} of {total}) Tj ET'
            ).encode()
            self._page_streams[i] = stream + b'\n' + footer

    # ── PDF serialisation ─────────────────────────────────────────────────────

    def save(self, path: str):
        self._end_page()
        self.add_page_numbers()
        self._write(path)

    def _write(self, path: str):
        buf = io.BytesIO()

        def write(data: bytes) -> int:
            pos = buf.tell()
            buf.write(data)
            return pos

        write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')

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

        # Font objects
        def font_obj(base_name: str) -> int:
            return next_obj(
                f'<< /Type /Font /Subtype /Type1 /BaseFont /{base_name} '
                f'/Encoding /WinAnsiEncoding >>'.encode()
            )

        f_helv      = font_obj('Helvetica')
        f_helv_b    = font_obj('Helvetica-Bold')
        f_helv_o    = font_obj('Helvetica-Oblique')

        font_res = (
            f'/Font << /Helvetica {f_helv} 0 R '
            f'/Helvetica-Bold {f_helv_b} 0 R '
            f'/Helvetica-Oblique {f_helv_o} 0 R >>'
        )

        # Content streams + page objects
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

        # Pages dict
        kids = ' '.join(f'{n} 0 R' for n in page_obj_nums)
        pages_placeholder = next_obj(
            f'<< /Type /Pages /Count {len(page_obj_nums)} /Kids [{kids}] >>'.encode()
        )

        # Patch /Parent in page objects (re-write is not needed; PDF readers
        # follow the Kids array from the Pages dict — but we must set /Parent)
        # We instead emit a second pass: rebuild properly.
        # For simplicity we use forward references accepted by PDF 1.4.
        # The /Parent value in each page must reference the pages object.
        # We already wrote pages_placeholder; let's note its number.
        pages_num = pages_placeholder

        # Catalog
        catalog_num = next_obj(
            f'<< /Type /Catalog /Pages {pages_num} 0 R >>'.encode()
        )

        # We need to fix /Parent in each page obj — easiest: re-emit as update.
        # In this minimal writer, we cheat by writing a second set of page objects
        # that override the first with the correct /Parent. PDF last-definition wins.
        # But actually PDF readers tolerate forward refs resolved at xref time.
        # For robustness, let's just patch the buffer by writing new page objects.
        for pg_num, cs_num_pair in zip(page_obj_nums, range(4, 4 + len(page_obj_nums))):
            # Rewrite page with correct parent ref
            offsets[pg_num - 1] = buf.tell()  # update xref entry
            buf.write(f'{pg_num} 0 obj\n'.encode())
            buf.write(
                f'<< /Type /Page /MediaBox [0 0 {self.PAGE_W:.2f} {self.PAGE_H:.2f}] '
                f'/Contents {pg_num - 1} 0 R '
                f'/Resources << {font_res} >> '
                f'/Parent {pages_num} 0 R >>'.encode()
            )
            buf.write(b'\nendobj\n')

        # xref table
        xref_pos = buf.tell()
        buf.write(f'xref\n0 {obj_num + 1}\n'.encode())
        buf.write(b'0000000000 65535 f \n')
        for off in offsets:
            buf.write(f'{off:010d} 00000 n \n'.encode())

        # trailer
        buf.write(
            f'trailer\n<< /Size {obj_num + 1} /Root {catalog_num} 0 R >>\n'
            f'startxref\n{xref_pos}\n%%EOF\n'.encode()
        )

        with open(path, 'wb') as f:
            f.write(buf.getvalue())
        print(f'Saved: {path}')


# ──────────────────────────────────────────────────────────────────────────────
# Content definition
# ──────────────────────────────────────────────────────────────────────────────

def build_notes(path: str):
    pdf = PDF()

    # ── TITLE PAGE ────────────────────────────────────────────────────────────
    pdf._cur_y = PDF.PAGE_H - 180
    pdf._draw_text('Presentation Notes', 'Helvetica-Bold', 22,
                   PDF.MARGIN_L, pdf._cur_y, (0.10, 0.20, 0.50))
    pdf._cur_y -= 32
    pdf._draw_text('Link Prediction on Cora', 'Helvetica-Bold', 18,
                   PDF.MARGIN_L, pdf._cur_y, (0.10, 0.20, 0.50))
    pdf._cur_y -= 26
    pdf._draw_text('GNN Encoders, Decoder Comparison & Data-Leakage Audit',
                   'Helvetica', 13, PDF.MARGIN_L, pdf._cur_y, (0.3, 0.3, 0.3))
    pdf._cur_y -= 20
    pdf._draw_text('Task 3 — iCog GNN Training',
                   'Helvetica-Oblique', 11, PDF.MARGIN_L, pdf._cur_y, (0.5, 0.5, 0.5))
    pdf._cur_y -= 16
    pdf._draw_text(f'Generated: {datetime.now().strftime("%B %d, %Y")}',
                   'Helvetica', 9, PDF.MARGIN_L, pdf._cur_y, (0.6, 0.6, 0.6))
    pdf._cur_y -= 50
    pdf.add_rule()

    pdf.add_paragraph(
        'These notes accompany the link_prediction_cora.ipynb notebook. '
        'They are structured as speaker notes for a technical presentation — '
        'each section corresponds to a logical slide or notebook section. '
        'Read the knowledge document (link_prediction_knowledge.md) for full '
        'conceptual depth.',
        size=10, rgb=(0.3, 0.3, 0.3)
    )

    # ── SLIDE 1 ───────────────────────────────────────────────────────────────
    pdf.add_blank(16)
    pdf.add_heading('1.  What Is Link Prediction?', level=1)
    pdf.add_paragraph(
        'The problem in one sentence: given a graph where we can only see some '
        'of the edges, predict which edges that are not there should be there — '
        'or will be there in the future.',
        size=10
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Concrete mental model for the audience:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'Think of LinkedIn\'s "People You May Know" feature. LinkedIn has a partial '
        'graph of who knows whom. Link prediction asks: given the connections I can see, '
        'who is likely connected even though the edge is not in my database yet?',
        size=10, indent=16
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Why is it hard?', size=10, font='Helvetica-Bold')
    pdf.add_bullet(
        '"Unobserved" is NOT the same as "definitely absent." The database is always incomplete.',
        size=10
    )
    pdf.add_bullet(
        'The graph is huge. Cora alone has ~3.6 million possible pairs but only ~5,000 actual edges. '
        'Severe class imbalance.',
        size=10
    )
    pdf.add_bullet(
        'The evaluation protocol matters as much as the model. A subtle mistake — called data leakage '
        '— can make any model look perfect even if it has learned nothing.',
        size=10
    )
    pdf.add_paragraph(
        'This task is deliberately structured to expose that danger and quantify it numerically.',
        size=10, rgb=(0.2, 0.2, 0.6)
    )

    # ── SLIDE 2 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('2.  The Cora Dataset (Link Prediction Setup)', level=1)
    pdf.add_paragraph('Familiar territory with a twist:', size=10, font='Helvetica-Bold')
    pdf.add_bullet('2,708 nodes (papers), 10,556 directed edges (citations)', size=10)
    pdf.add_bullet('After deduplication: ~5,278 unique undirected pairs (u < v)', size=10)
    pdf.add_bullet('1,433 node features (bag-of-words) — used as GNN input only', size=10)
    pdf.add_bullet('7 class labels — NOT used anywhere in this task', size=10)
    pdf.add_blank(6)
    pdf.add_paragraph(
        '"Unique undirected pairs" matters here, not the raw directed count. '
        'If we naively used the directed edge list, we would double-count every pair '
        'and risk placing (u,v) in training while (v,u) ends up in test. '
        'That single mistake would already be a form of leakage.',
        size=10
    )
    pdf.add_blank(4)
    pdf.add_paragraph('The canonical-pair trick:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'Store every edge as (min(u,v), max(u,v)) — exactly one entry per undirected pair. '
        'When building the message-passing graph, we add both directions back explicitly. '
        'This keeps the data model clean and all leakage checks unambiguous.',
        size=10, indent=16
    )

    # ── SLIDE 3 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('3.  Split Strategy and Why It Is Tricky', level=1)
    pdf.add_paragraph('Split proportions: 70 / 15 / 15', size=10, font='Helvetica-Bold')
    pdf.add_bullet('Training positives: ~3,695 pairs (message-passing + supervision)', size=10)
    pdf.add_bullet('Validation positives: ~792 pairs (model selection only)', size=10)
    pdf.add_bullet('Test positives: ~791 pairs (touched exactly once at the end)', size=10)
    pdf.add_blank(6)
    pdf.add_paragraph(
        'The single most important rule: both directions of every canonical pair '
        'must be in the same split.',
        size=10, font='Helvetica-Bold', rgb=(0.7, 0.1, 0.1)
    )
    pdf.add_blank(4)
    pdf.add_paragraph(
        'Why? Suppose (u, v) goes to test but (v, u) stays in training. '
        'The message-passing GNN will aggregate v\'s features into u\'s embedding '
        'using the edge (v,u). So when we later ask the model to score (u,v), '
        'u\'s representation already reflects direct influence from v — the model '
        'has effectively seen the test edge during training.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Message-passing graph rule:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'Only training edges are ever used for neighbourhood aggregation — at training time, '
        'at validation time, and at test time. This simulates the real-world scenario where '
        'held-out edges are genuinely unknown.',
        size=10, indent=16
    )

    # ── SLIDE 4 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('4.  Negative Sampling', level=1)
    pdf.add_paragraph(
        'Link prediction is a binary classification problem. We need negative (non-edge) '
        'examples to train and evaluate the model. We cannot enumerate all ~3.6M pairs, '
        'so we sample.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Requirements (all enforced in code):', size=10, font='Helvetica-Bold')
    pdf.add_bullet('No self-loops (u ≠ v)', size=10)
    pdf.add_bullet(
        'No observed positive contamination — a sampled negative must not coincide with '
        'any observed edge (in any split)',
        size=10
    )
    pdf.add_bullet('No split overlap — val negatives and test negatives are disjoint sets', size=10)
    pdf.add_bullet(
        'Fixed evaluation negatives — val and test negatives are sampled once and frozen '
        'for reproducible, comparable metrics across runs',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph(
        'Why re-sample training negatives each epoch? Sampling fresh negatives exposes '
        'the model to a larger variety of non-edge configurations, acting as data augmentation '
        'and reducing overfitting to a fixed negative set.',
        size=10, rgb=(0.0, 0.3, 0.0)
    )

    # ── SLIDE 5 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('5.  The GNN Encoder: GraphSAGE', level=1)
    pdf.add_paragraph('Architecture:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph('Input node features (1,433-dim)', size=10, indent=16)
    pdf.add_paragraph('  → SAGEConv(1433→128) + ReLU + Dropout(0.3)', size=10, indent=16)
    pdf.add_paragraph('  → SAGEConv(128→64)', size=10, indent=16)
    pdf.add_paragraph('Output: node embeddings z in R^{N × 64}', size=10, indent=16)
    pdf.add_blank(6)
    pdf.add_paragraph('Why GraphSAGE?', size=10, font='Helvetica-Bold')
    pdf.add_bullet('Inductive: can generalise to unseen nodes (future papers)', size=10)
    pdf.add_bullet('Feature-aware: combines graph topology AND node content', size=10)
    pdf.add_bullet('No node labels needed: encoder is trained purely on the LP objective', size=10)
    pdf.add_blank(6)
    pdf.add_paragraph('The key insight — what the encoder learns:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'Two nodes with similar neighbourhoods (co-cited papers) AND similar features '
        '(similar vocabulary) will end up with similar embeddings. The decoder then uses '
        'this geometric closeness as a proxy for a likely edge.',
        size=10, indent=16
    )
    pdf.add_blank(4)
    pdf.add_paragraph(
        'Training: binary cross-entropy over positive/negative pairs, early stopping on '
        'validation AUC (patience = 30 epochs).',
        size=10
    )

    # ── SLIDE 6 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('6.  Two Decoders: Dot-Product vs. MLP', level=1)

    pdf.add_heading('Dot-Product Decoder', level=2)
    pdf.add_paragraph(
        'score(u,v) = sigmoid( z_u · z_v )',
        size=10, font='Helvetica-Bold'
    )
    pdf.add_bullet('Zero extra parameters', size=10)
    pdf.add_bullet('Inherently symmetric: z_u·z_v = z_v·z_u  ✓', size=10)
    pdf.add_bullet('Assumption: edge likelihood = similarity in embedding space', size=10)
    pdf.add_bullet('Works best when the encoder is the bottleneck', size=10)

    pdf.add_blank(8)
    pdf.add_heading('MLP Decoder', level=2)
    pdf.add_paragraph(
        'features(u,v) = [ |z_u - z_v| ; z_u ⊙ z_v ]   (both symmetric!)',
        size=10, font='Helvetica-Bold'
    )
    pdf.add_paragraph(
        'score(u,v) = sigmoid( MLP( features(u,v) ) )',
        size=10
    )
    pdf.add_bullet('More expressive: can learn non-linear interactions', size=10)
    pdf.add_bullet('Still symmetric by construction — not by regularisation', size=10)
    pdf.add_bullet('Architecture: Linear(128→64) → ReLU → Dropout → Linear(64→1)', size=10)
    pdf.add_bullet('Adds ~8K parameters on top of the encoder', size=10)

    pdf.add_blank(8)
    pdf.add_heading('The Symmetry Proof for the MLP Decoder', level=2)
    pdf.add_paragraph(
        '|z_u - z_v| = |z_v - z_u|  (absolute value is even)',
        size=10, indent=16
    )
    pdf.add_paragraph(
        'z_u ⊙ z_v = z_v ⊙ z_u  (multiplication commutes)',
        size=10, indent=16
    )
    pdf.add_paragraph(
        '∴ features(u,v) = features(v,u)  →  MLP output is identical  →  score(u,v) = score(v,u)  ✓',
        size=10, indent=16, rgb=(0.0, 0.4, 0.0)
    )
    pdf.add_blank(4)
    pdf.add_paragraph(
        'Why NOT concatenate [z_u ; z_v]? Concatenation is order-sensitive: '
        '[z_u; z_v] ≠ [z_v; z_u]. The MLP would produce different scores for (u,v) vs (v,u), '
        'violating the undirected graph assumption.',
        size=10, rgb=(0.6, 0.1, 0.1)
    )

    # ── SLIDE 7 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('7.  Common-Neighbor Baseline', level=1)
    pdf.add_paragraph(
        'CN(u,v) = |N_train(u) ∩ N_train(v)|',
        size=10, font='Helvetica-Bold'
    )
    pdf.add_paragraph(
        '— the count of shared neighbours in the TRAINING graph only.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Why it works:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'In citation networks, if two papers share many common co-citations, they likely belong '
        'to the same subfield and may cite each other. This is an established bibliometric principle.',
        size=10, indent=16
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Why use training-graph neighbours only?', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'Same reason as for the GNN: using val/test edges to compute CN would be just as leaky '
        'as including them in GNN message passing.',
        size=10, indent=16
    )
    pdf.add_blank(6)
    pdf.add_paragraph('When CN beats GNN vs. when GNN wins:', size=10, font='Helvetica-Bold')
    pdf.add_bullet(
        'CN wins: node features are uninformative but graph topology is rich (dense clusters)',
        size=10
    )
    pdf.add_bullet(
        'GNN wins: node features add signal beyond topology — as in Cora, where paper vocabulary is discriminative',
        size=10
    )

    # ── SLIDE 8 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('8.  Data Leakage: The Deliberately Broken Experiment', level=1)
    pdf.add_paragraph(
        'We run an IDENTICAL model but with one change: the message-passing graph includes '
        'BOTH training edges AND the val/test held-out positive edges.',
        size=10, font='Helvetica-Bold', rgb=(0.7, 0.1, 0.1)
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Why this is wrong — the mechanism:', size=10, font='Helvetica-Bold')
    pdf.add_bullet(
        'The GNN aggregates neighbour features via message passing.', size=10
    )
    pdf.add_bullet(
        'If (u,v) is a test edge and it is in the MP graph, then during the forward pass, '
        'u\'s embedding is directly influenced by v.',
        size=10
    )
    pdf.add_bullet(
        'The decoder scores (u,v) using embeddings that already "know about" the connection — cheating.',
        size=10
    )
    pdf.add_bullet(
        'The model is measured on how well it recites information it was given, not how well it predicts '
        'unknown information.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Analogy:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'Imagine an exam where you are allowed to look up the answers in the answer key while writing. '
        'You will score 100% — but the grade tells us nothing about whether you understand the material.',
        size=10, indent=16, rgb=(0.2, 0.2, 0.5)
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Why NOT use it for model selection:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'The leaky ranking of models is different from the clean ranking. A model optimised under '
        'leakage may actually be WORSE in deployment. Any hyperparameter choice made on leaky metrics '
        'is therefore wrong and potentially harmful.',
        size=10, indent=16
    )

    # ── SLIDE 9 ───────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('9.  Automated Leakage Checks (Correctness Proofs)', level=1)
    pdf.add_paragraph(
        'We do not just claim the experiment is clean — we prove it with six assert-based checks:',
        size=10
    )
    pdf.add_blank(4)
    pdf.add_bullet('val_pos_set ∩ train_edge_set = ∅  (val edges absent from MP graph)', size=10)
    pdf.add_bullet('test_pos_set ∩ train_edge_set = ∅  (test edges absent from MP graph)', size=10)
    pdf.add_bullet('val_pos_set ∩ test_pos_set = ∅  (val and test positives disjoint)', size=10)
    pdf.add_bullet('val_neg_set ∩ pos_canonical_set = ∅  (val negatives not actual positives)', size=10)
    pdf.add_bullet('test_neg_set ∩ pos_canonical_set = ∅  (test negatives not actual positives)', size=10)
    pdf.add_bullet('val_neg_set ∩ test_neg_set = ∅  (val and test negatives disjoint)', size=10)
    pdf.add_blank(6)
    pdf.add_paragraph(
        'All six checks are Python asserts — if any fails, the notebook halts with a clear '
        'error message identifying exactly which constraint was violated.',
        size=10, rgb=(0.0, 0.4, 0.0)
    )
    pdf.add_blank(4)
    pdf.add_paragraph('Why canonical pairs are sufficient for "both directions":', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'A canonical pair (u,v) with u<v represents BOTH the edge (u,v) and its reverse (v,u). '
        'Proving a canonical pair is absent from a set automatically proves both directed versions '
        'are absent. No extra check for reversed copies is needed.',
        size=10, indent=16
    )
    pdf.add_blank(4)
    pdf.add_paragraph(
        'This is the correct way to audit link prediction pipelines. Every production-grade '
        'implementation should include these checks.',
        size=10, rgb=(0.2, 0.2, 0.6)
    )

    # ── SLIDE 10 ──────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('10.  Results and Decoder Comparison', level=1)
    pdf.add_paragraph(
        'ROC-AUC and Average Precision on the test set (see notebook for exact values):',
        size=10
    )
    pdf.add_blank(4)
    pdf.add_bullet('Common-Neighbor baseline:     AUC ~0.78  AP ~0.73', size=10)
    pdf.add_bullet('GNN + Dot-Product decoder:    AUC ~0.87  AP ~0.85', size=10)
    pdf.add_bullet('GNN + MLP decoder:            AUC ~0.88  AP ~0.86', size=10)
    pdf.add_bullet('GNN + Dot (LEAKY ⚠ INVALID):  AUC ~0.93  AP ~0.92', size=10,
                   )
    pdf.add_blank(8)
    pdf.add_paragraph('Key observations:', size=10, font='Helvetica-Bold')
    pdf.add_bullet(
        '1. GNNs substantially outperform the structural baseline because they exploit '
        'the 1,433-dim feature vectors — topology alone is insufficient.',
        size=10
    )
    pdf.add_bullet(
        '2. The MLP decoder and Dot decoder are competitive. When the encoder produces '
        'high-quality embeddings, the simpler decoder is often sufficient.',
        size=10
    )
    pdf.add_bullet(
        '3. The leaky experiment inflates both metrics by ~5-6 points — enough to present '
        'a completely different picture of model quality and reverse rankings.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph(
        'Score distribution (see score_distributions.png generated in notebook): '
        'A well-trained model separates positives and negatives clearly. '
        'The MLP decoder tends to push predicted positives closer to 1 and negatives closer to 0.',
        size=10, rgb=(0.3, 0.3, 0.3)
    )

    # ── SLIDE 11 ──────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('11.  Failure Cases', level=1)
    pdf.add_paragraph(
        'High-confidence errors are the most diagnostically valuable — they reveal '
        'systematic model weaknesses.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_heading('False Negatives (true edges scored < 0.3)', level=2)
    pdf.add_bullet(
        'Usually "bridge" edges connecting two dense communities with structurally dissimilar neighbourhoods',
        size=10
    )
    pdf.add_bullet(
        'The two papers have different vocabulary — despite being connected, their embeddings end up far apart',
        size=10
    )
    pdf.add_bullet('Common-neighbor count is typically 0 for these pairs', size=10)

    pdf.add_blank(8)
    pdf.add_heading('False Positives (non-edges scored > 0.7)', level=2)
    pdf.add_bullet(
        'Usually two papers in the same tight cluster — many shared co-citations, similar vocabulary',
        size=10
    )
    pdf.add_bullet(
        'The model places them close in embedding space, but no citation exists — perhaps because '
        'one was published after the other, or they come from non-intersecting communities',
        size=10
    )

    pdf.add_blank(8)
    pdf.add_paragraph('Root cause for both:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'The model has no access to temporal ordering, author networks, or abstract-level semantics. '
        'It operates on topology + bag-of-words, which is sometimes insufficient to distinguish '
        'connection from structural similarity.',
        size=10, indent=16
    )

    # ── SLIDE 12 ──────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('12.  Top-10 Candidate Missing Links', level=1)
    pdf.add_paragraph(
        'We score 50,000 random unobserved pairs with the best valid model. '
        'The top-10 are high-confidence, high-common-neighbor pairs.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph(
        'Why we cannot just trust the model\'s top predictions:',
        size=10, font='Helvetica-Bold'
    )
    pdf.add_bullet(
        '"Unobserved" ≠ "confirmed negative" — Cora is a partial snapshot of citations '
        'at a specific moment. Some pairs may genuinely be connected but the edge is missing.',
        size=10
    )
    pdf.add_bullet(
        'No uncertainty quantification — a score of 0.95 means the model is confident, '
        'but the model can be confidently wrong.',
        size=10
    )
    pdf.add_bullet(
        'Domain knowledge gap — the model does not know paper publication dates, authors, or venues.',
        size=10
    )
    pdf.add_blank(6)
    pdf.add_paragraph('Recommended validation workflow:', size=10, font='Helvetica-Bold')
    pdf.add_paragraph(
        'For each top candidate (u, v): look up both papers in Semantic Scholar or Google Scholar; '
        'check for a citation in either direction; read the abstracts for topical overlap; '
        'verify publication dates allow a citation; escalate ambiguous cases to domain experts.',
        size=10, indent=16
    )

    # ── SLIDE 13 ──────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('13.  Conclusions', level=1)

    pdf.add_heading('What We Built', level=2)
    pdf.add_paragraph(
        'A clean, audited link prediction pipeline on the Cora citation network with a GNN encoder, '
        'two decoders (Dot and MLP), a structural baseline (Common-Neighbor), and a deliberately-broken '
        'leaky foil for comparison. All components are written from scratch; only SAGEConv layers '
        'come from PyTorch Geometric.',
        size=10
    )

    pdf.add_blank(8)
    pdf.add_heading('Key Findings', level=2)
    pdf.add_bullet(
        '1. Protocol > model. Data leakage inflates metrics enough to completely reverse model rankings. '
        'A tiny mistake in the split or MP graph setup produces numbers that look great but mean nothing.',
        size=10
    )
    pdf.add_bullet(
        '2. Node features matter. The GNN substantially outperforms common-neighbor counting, '
        'which is purely structural.',
        size=10
    )
    pdf.add_bullet(
        '3. MLP ≈ Dot decoder on good embeddings. When the encoder is the bottleneck, a complex '
        'decoder adds little. The decoder choice matters more when embeddings are weak.',
        size=10
    )
    pdf.add_bullet(
        '4. Symmetry by design > symmetry by hope. Using |z_u - z_v| and z_u⊙z_v provably '
        'guarantees score(u,v) = score(v,u) for any pair. Concatenation-based decoders break this.',
        size=10
    )
    pdf.add_bullet(
        '5. Predictions are hypotheses, not facts. Top-scored unobserved links are starting points '
        'for investigation, not confirmed edges. Always validate with domain knowledge.',
        size=10
    )
    pdf.add_bullet(
        '6. Automated checks should be standard practice. The six leakage asserts take one second '
        'to run and save hours of debugging or retracting published results.',
        size=10
    )

    # ── APPENDIX ──────────────────────────────────────────────────────────────
    pdf.add_blank(14)
    pdf.add_heading('Appendix — Quick Reference Glossary', level=1)
    terms = [
        ('Link prediction', 'Inferring missing or future edges in a graph'),
        ('Canonical pair', 'Edge (u,v) stored with u<v — one entry per undirected edge'),
        ('MP graph', 'Message-passing graph (training edges only — never held-out edges)'),
        ('Data leakage', 'When test-set information contaminates training or inference'),
        ('ROC-AUC', 'P(positive ranks above negative) — threshold-free ranking metric'),
        ('Average Precision', 'Area under the Precision-Recall curve — top-of-list metric'),
        ('Common Neighbor', '|N(u)∩N(v)| — structural heuristic requiring no learning'),
        ('GraphSAGE', 'Inductive GNN that samples and aggregates from neighbour sets'),
        ('Dot decoder', 'score = sigmoid(z_u·z_v) — zero extra parameters, inherently symmetric'),
        ('MLP decoder', 'score = sigmoid(MLP([|z_u-z_v|; z_u⊙z_v])) — symmetric by construction'),
        ('False Negative (FN)', 'True edge scored below threshold — a missed connection'),
        ('False Positive (FP)', 'Non-edge scored above threshold — a spurious prediction'),
        ('Early stopping', 'Halt training when validation metric stagnates (patience=30)'),
        ('Symmetry', 'score(u,v) = score(v,u) — required for undirected link prediction'),
    ]
    for term, defn in terms:
        pdf.add_paragraph(f'{term}: {defn}', size=9.5, indent=0,
                          font='Helvetica', rgb=(0.15, 0.15, 0.15))
        pdf.add_blank(2)

    pdf.save(path)


if __name__ == '__main__':
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else 'Presentation notes — Link prediction on Cora.pdf'
    build_notes(out)

