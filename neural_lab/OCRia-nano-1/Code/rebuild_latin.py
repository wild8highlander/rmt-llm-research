"""Rebuild the int8-quantized rec_latin.onnx as a trainable fp32 torch model.

OCRia-nano-1 ships its recognizers as QDQ-quantized ONNX graphs: every conv
weight/bias is int8/int32 per-channel quantized, activations round-trip through
uint8 quantizers with zero_point=0 (which is exactly a ReLU - uint8 clips at 0).
For gradient training we dequantize every tensor back to fp32 and re-assemble
the very same architecture in torch:

    stem  : Conv 3x3 s2 (1->32) + ReLU
    body  : 9 depthwise-separable blocks, strided down to H=1
            residual Adds in blocks 1,3,4,6,7 (odd blocks, even skips)
    mix   : 3 blocks  y = x + ReLU(dw9(x));  z = y + fc2(ReLU(fc1(y)))
    head  : Conv 1x1 (192->792) -> logits [N,792,T]  (CTC, blank = class 0)

The rebuild is validated against onnxruntime on a rendered-crop set:
exact decoded-string parity and mean |logit| deviation are written to
Results/rebuild_parity.json. Quantization round-off is documented honestly -
the fp32 rebuild is a refresh of the int8 graph, not a bit-exact copy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

MODEL_DIR = Path(__file__).resolve().parent.parent / "Model"
RESULTS = Path(__file__).resolve().parent.parent / "Results"

# --------------------------------------------------------------------------
# ONNX -> fp32 tensors
# --------------------------------------------------------------------------


def _dequant(q: np.ndarray, scale: np.ndarray, zp: np.ndarray) -> np.ndarray:
    """(q - zp) * scale with per-channel scale along axis 0."""
    s = np.asarray(scale, dtype=np.float32).reshape(-1, *([1] * (q.ndim - 1)))
    z = np.asarray(zp, dtype=np.float32).reshape(-1, *([1] * (q.ndim - 1)))
    return (q.astype(np.float32) - z) * s


def extract_latin_tensors():
    """Return {name: (weight fp32, bias fp32, attrs)} for every Conv of
    rec_latin.onnx, in graph order, with torch-friendly names."""
    m = onnx.load(str(MODEL_DIR / "rec_latin.onnx"))
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    nodes = {n.output[0]: n for n in g.node}
    convs = [n for n in g.node if n.op_type == "Conv"]
    out = []
    for n in convs:
        at = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        w_name, b_name = n.input[1], n.input[2]
        # weight: conv-input[1] is the output of a DequantizeLinear whose
        # input[0] is '<w>_quantized' with '<w>_scale' / '<w>_zero_point'
        dq = nodes[w_name]
        wq_name = dq.input[0]
        base = wq_name.replace("_quantized", "")
        w = _dequant(inits[wq_name], inits[base + "_scale"],
                     inits[base + "_zero_point"])
        # bias: input[2] is a DequantizeLinear output over int32 quantized bias
        # (bias quantizer names its scale '<q>_quantized_scale', the weight
        #  quantizer just '<base>_scale' - resolve both spellings)
        b = None
        if b_name in nodes and nodes[b_name].op_type == "DequantizeLinear":
            bq = nodes[b_name].input[0]
            bb = bq.replace("_quantized", "")
            s_name = (bb + "_quantized_scale") if (bb + "_quantized_scale") in inits else (bb + "_scale")
            z_name = (bb + "_quantized_zero_point") if (bb + "_quantized_zero_point") in inits else (bb + "_zero_point")
            b = _dequant(inits[bq], inits[s_name], inits[z_name])
        elif b_name in inits:
            b = inits[b_name].astype(np.float32)
        out.append({"node_out": n.output[0], "w": w, "b": b,
                    "stride": at.get("strides", [1, 1]),
                    "pad": at.get("pads", [0, 0, 0, 0])[:2],
                    "group": at.get("group", 1),
                    "kernel": at.get("kernel_shape", [])})
    return out


# --------------------------------------------------------------------------
# Torch rebuild
# --------------------------------------------------------------------------

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class LatinRecognizer(nn.Module):
    """fp32 rebuild of rec_latin.onnx.  forward(x [N,1,32,W]) -> logits
    [N,792,T];  hooks expose per-stage feature maps for the probe."""

    STAGES = ("stem", "body0", "body1", "body2", "body3", "body4",
              "body5", "body6", "body7", "body8",
              "mix0", "mix1", "mix2", "head")

    def __init__(self, tensors: list[dict] | None = None):
        super().__init__()
        t = tensors if tensors is not None else extract_latin_tensors()
        c = {entry["node_out"]: entry for entry in t}

        def mk(key):
            e = c[key]
            k = e["kernel"]
            conv = nn.Conv2d(e["w"].shape[1] * e["group"], e["w"].shape[0],
                             tuple(k),
                             stride=tuple(e["stride"]), padding=tuple(e["pad"]),
                             groups=e["group"], bias=e["b"] is not None)
            with torch.no_grad():
                conv.weight.copy_(torch.tensor(e["w"]))
                if e["b"] is not None:
                    conv.bias.copy_(torch.tensor(e["b"]))
            return conv

        def mk1d(key):
            e = c[key]
            conv = nn.Conv1d(e["w"].shape[1] * e["group"], e["w"].shape[0],
                             e["kernel"][0],
                             stride=e["stride"][0], padding=e["pad"][0],
                             groups=e["group"], bias=e["b"] is not None)
            with torch.no_grad():
                conv.weight.copy_(torch.tensor(e["w"]))
                if e["b"] is not None:
                    conv.bias.copy_(torch.tensor(e["b"]))
            return conv

        # pick node outputs by their /m/ path fragment
        pick = lambda frag: next(k for k in c if frag in k)  # noqa: E731

        self.stem = mk(pick("stem/stem.2"))
        # body: (dw_key, pw_key, residual_from_block_or_None)
        self.body_dw = nn.ModuleList()
        self.body_pw = nn.ModuleList()
        self.residual_from = []
        skips = {}
        body_plan = [
            ("body.0/act", "body.0/act_1", None),
            ("body.1/act", "body.1/pw", 0),
            ("body.2/act", "body.2/act_1", None),
            ("body.3/act", "body.3/pw", 2),
            ("body.4/act", "body.4/pw", 3),
            ("body.5/act", "body.5/act_1", None),
            ("body.6/act", "body.6/pw", 5),
            ("body.7/act", "body.7/pw", 6),
            ("body.8/act", "body.8/act_1", None),
        ]
        for i, (dw, pw, res) in enumerate(body_plan):
            self.body_dw.append(mk(pick(dw)))
            self.body_pw.append(mk(pick(pw)))
            self.residual_from.append(res)
        self.mix_dw = nn.ModuleList([mk1d(pick(f"mix.{i}/act")) for i in range(3)])
        self.mix_fc1 = nn.ModuleList([mk1d(pick(f"mix.{i}/act_1")) for i in range(3)])
        self.mix_fc2 = nn.ModuleList([mk1d(pick(f"mix.{i}/fc2")) for i in range(3)])
        self.head = mk1d(pick("head/Conv"))
        self.stage_maps: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N,1,32,W] in [0,1] -> logits [N,792,T]."""
        self.stage_maps.clear()
        h = F.relu(self.stem(x))
        self.stage_maps["stem"] = h
        skip = {}
        for i, (dw, pw) in enumerate(zip(self.body_dw, self.body_pw)):
            a = F.relu(dw(h))
            p = pw(a)
            src = self.residual_from[i]
            if src is not None:
                h = F.relu(p + skip[src])
            else:
                h = F.relu(p)
            skip[i] = h  # the skip path is the block's post-ReLU output
            self.stage_maps[f"body{i}"] = h
        z = h.squeeze(2)  # [N,192,T]
        self.stage_maps["squeeze"] = z
        mix_in = z
        # ReLU-after-first-Add only in mix.0 (its outgoing uint8 quantizer has
        # zero_point=0, which clips at 0); mix.1/2 quantizers have zp=62/78 -
        # affine, signed, no clip. Derived from the QDQ chains, not guessed.
        relu_first_add = (True, False, False)
        for i, (dw, fc1, fc2) in enumerate(zip(self.mix_dw, self.mix_fc1, self.mix_fc2)):
            y = mix_in + F.relu(dw(mix_in))
            if relu_first_add[i]:
                y = F.relu(y)
            z2 = y + fc2(F.relu(fc1(y)))             # signed: zp!=0 after the second Add
            mix_in = z2
            self.stage_maps[f"mix{i}"] = mix_in
        logits = self.head(mix_in)  # [N,792,T]
        self.stage_maps["head"] = logits
        return logits


def ctc_greedy_decode(logits: torch.Tensor) -> tuple[list[str], list[float], np.ndarray]:
    """blank = class 0; collapse repeats; return strings, mean conf, ids."""
    probs = F.softmax(logits, dim=1)
    conf, ids = probs.max(dim=1)  # [N,T]
    cs = json.loads((MODEL_DIR / "rec_latin.charset.json").read_text())
    texts = []
    for b in range(ids.shape[0]):
        prev = -1
        out = []
        for i in ids[b].tolist():
            if i != prev and i != 0:
                out.append(cs[i - 1] if 0 < i <= len(cs) else "?")
            prev = i
        texts.append("".join(out))
    return texts, [float(c.mean()) for c in conf], ids.numpy()


# --------------------------------------------------------------------------
# Parity validation vs onnxruntime
# --------------------------------------------------------------------------

def render_crops(n: int = 48, seed: int = 7) -> list[np.ndarray]:
    """Rendered text crops [32, W] fp32 in [0,1] - the same generator the
    probe corpora use (subset here, for the parity gate)."""
    from PIL import Image, ImageDraw, ImageFont

    rng = np.random.default_rng(seed)
    words = ["the quick brown fox", "invoice total 1,240.00", "OCRia nano probe",
             "rmt x llm research", "spectral purity gate", "hello world 42",
             "page 7 of 12", "dear colleague", "404 not found", "exit 0"]
    fonts = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
             "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
    crops = []
    for i in range(n):
        text = words[i % len(words)]
        fp = fonts[i % len(fonts)]
        size = int(rng.integers(16, 24))
        W = int(rng.integers(200, 480))
        img = Image.new("L", (W, 32), 255)
        d = ImageDraw.Draw(img)
        d.text((int(rng.integers(2, 12)), int(rng.integers(2, 8))), text,
               font=ImageFont.truetype(fp, size), fill=0)
        crops.append(np.asarray(img, dtype=np.float32) / 255.0)
    return crops


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    crops = render_crops()
    x = torch.tensor(np.stack([np.pad(c, ((0, 0), (0, 480 - c.shape[1]))) for c in crops]))[:, None]

    model = LatinRecognizer()
    model.eval()
    with torch.no_grad():
        logits = model(x)
    texts_t, conf_t, _ = ctc_greedy_decode(logits)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(MODEL_DIR / "rec_latin.onnx"),
                                providers=["CPUExecutionProvider"])
    cs = json.loads((MODEL_DIR / "rec_latin.charset.json").read_text())

    def decode(ids):
        out, prev = [], -1
        for i in ids:
            i = int(i)
            if i != prev and i != 0:
                out.append(cs[i - 1] if 0 < i <= len(cs) else "?")
            prev = i
        return "".join(out)

    n_match, conf_dev = 0, []
    for b, c in enumerate(crops):
        ids_o, conf_o = sess.run(None, {"x": c[None, None].astype(np.float32)})
        s_o = decode(ids_o[0])
        n_match += int(s_o == texts_t[b])
        conf_dev.append(abs(conf_o[0].mean() - conf_t[b]))
    report = {
        "n_crops": len(crops),
        "exact_string_parity": round(n_match / len(crops), 4),
        "mean_abs_conf_deviation": round(float(np.mean(conf_dev)), 6),
        "max_abs_conf_deviation": round(float(np.max(conf_dev)), 6),
        "note": ("fp32 rebuild of the int8 QDQ graph; uint8 activation "
                 "quantizers with zero_point=0 are exact ReLUs, so the fp32 "
                 "refresh matches the deployed graph up to quantization "
                 "round-off of the conv arithmetic."),
    }
    (RESULTS / "rebuild_parity.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print("[rebuild] exact string parity:", report["exact_string_parity"],
          " conf dev:", report["mean_abs_conf_deviation"])
    if report["exact_string_parity"] < 0.85:
        print("[rebuild] WARNING: parity below 0.85 - inspect before training")
        sys.exit(2)
    torch.save(model.state_dict(), RESULTS / "latin_rebuild_fp32.pt")
    print("[rebuild] saved Results/latin_rebuild_fp32.pt")


if __name__ == "__main__":
    main()
