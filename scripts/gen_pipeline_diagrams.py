#!/usr/bin/env python3
"""SpherIQ pipeline — diagrams library (graphviz DOT auto-layout)."""

from diagrams import Diagram, Cluster, Edge
from diagrams.generic.blank import Blank
from diagrams.custom import Custom
import os

OUT = "figures"
os.makedirs(OUT, exist_ok=True)

graph_attr = {
    "fontsize": "18",
    "bgcolor": "white",
    "rankdir": "TB",
    "splines": "ortho",
    "pad": "0.6",
    "nodesep": "0.35",
    "ranksep": "0.45",
    "dpi": "200",
    "compound": "true",
    "fontname": "Helvetica",
}

node_attr = {
    "fontsize": "10",
    "fontname": "Helvetica",
    "margin": "0.18,0.10",
    "width": "1.6",
    "height": "0.55",
    "style": "rounded,filled",
    "fillcolor": "white",
    "penwidth": "2",
}

C_AUG   = '#1565C0'
C_PRE   = '#E65100'
C_FEAT  = '#6A1B9A'
C_ENC   = '#283593'
C_FUSE  = '#4E342E'
C_PRED  = '#AD1457'
C_INFER = '#00838F'

with Diagram(
    "SpherIQ Pipeline  —  Omnidirectional IQA",
    filename=os.path.join(OUT, "spheriq_pipeline_diagrams"),
    direction="TB",
    show=False,
    graph_attr=graph_attr,
    node_attr=node_attr,
    outformat="svg",
):
    inp = Blank("ERP Input\n(B, 3, H, W)")

    with Cluster("Data Augmentation (training)"):
        rot = Blank("Spherical 3D\nRotation")
        fused = Blank("Fused Single-Pass\nCubemap Projection")
        art = Blank("Synthetic\nArtifacts")

    with Cluster("Preprocessing"):
        cube = Blank("Cubemap\n6 faces (12 stereo)")
        patch = Blank("Multi-Scale\nPatch Extraction")

    with Cluster("Patch Embedding"):
        cnn = Blank("CNN Backbone\nStdConv → GN → Bottleneck")

    with Cluster("Geometric Embeddings"):
        rope = Blank("3D Rotary PE\n(continuous coords)")
        face = Blank("Face Emb\n(6 × 384-d)")
        scale = Blank("Scale Emb\n(3 × 384-d)")

    with Cluster("Transformer Encoder\n(14 layers)"):
        enc = Blank("LN → MHA(4h) → DropPath\nLN → MLP(1152) → DropPath")

    with Cluster("Face Aggregation"):
        meta = Blank("Meta-Transformer\n(3 layers + ERP bias)")
        agg = Blank("[AGG] Token\n→ Global Repr.")

    with Cluster("Prediction"):
        heads = Blank("Quality Heads\n(mean + log-variance)")
        tta = Blank("TTA Ensemble\n(0/90/180/270°)")
        stereo = Blank("Stereo Fusion\n(avg L/R)")

    out = Blank("Quality Score")

    # Edges
    inp >> rot
    rot >> fused
    fused >> art
    art >> Edge(label="p=0.15\ndropout", color=C_AUG) >> cube
    cube >> Edge(color=C_PRE) >> patch
    patch >> Edge(color=C_FEAT) >> cnn
    cnn >> Edge(label="(N, 384)", color=C_FEAT) >> rope
    rope >> face >> scale
    scale >> Edge(color=C_ENC) >> enc
    Blank("6× CLS Tokens\n(1 per face)") >> Edge(color=C_ENC) >> enc
    enc >> Edge(color=C_FUSE) >> meta
    meta >> agg
    agg >> Edge(color=C_PRED) >> heads
    heads >> Edge(label="eval", color=C_PRED) >> tta
    tta >> Edge(color=C_INFER) >> stereo
    stereo >> Edge(color=C_INFER) >> out

print("Saved diagrams output")
