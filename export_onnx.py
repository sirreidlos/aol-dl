import torch
import argparse

from torch.export import Dim
from models import ResNet, Generator, ResNetConfig, GeneratorConfig

import onnxruntime as ort
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="Path to the input file")
parser.add_argument("output", type=str, help="Output path")
args = parser.parse_args()

checkpoint = torch.load(args.file, weights_only=False)
if checkpoint.get("model"):
    config_dict = checkpoint.get("config", {})
    config = ResNetConfig(**config_dict)
    weights = checkpoint["model"]
    model = ResNet(config)
    model.load_state_dict(weights)
else:
    config_dict = checkpoint.get("generator_config", {})
    config = GeneratorConfig(**config_dict)
    weights = checkpoint["generator"]
    model = Generator(config)
    model.load_state_dict(weights)


model.eval()
dummy_input = torch.randn(1, 3, 96, 96)

# torch.onnx.export(
#     model,
#     dummy_input,
#     args.output,
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_shapes={"x": {0: Dim.DYNAMIC, 2: Dim.DYNAMIC, 3: Dim.DYNAMIC}},
#     export_params=True,  # Explicitly embed parameters
#     do_constant_folding=True,
#     opset_version=17,  # Use a stable opset version
#     **{"save_as_external_data": False},  # Force single file
# )
torch.onnx.export(
    model,
    dummy_input,
    args.output,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={  # Use dynamic_axes for legacy API
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
    opset_version=17,
    dynamo=False,  # Force legacy exporter
    do_constant_folding=True,
)


session = ort.InferenceSession(args.output)
dummy_input_np = dummy_input.numpy()
outputs = session.run(None, {"input": dummy_input_np})
print(f"ONNX output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
