"""TFLite export script for NAV-X 3.0 models.

Exports all trained models to TFLite format with INT8 quantization.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


from aeronavis.config import Config, get_config
from aeronavis.models.velocity import VelocityModel, build_velocity_model
from aeronavis.models.pseudo_odo import PseudoOdoModel, build_pseudo_odo_model
from aeronavis.models.slip import SlipClassifier, build_slip_model
from aeronavis.models.visual_odo import VisualOdoNet, build_visual_odo
from aeronavis.models.texture_gate import TextureGate, build_texture_gate
from aeronavis.predict.model import PredictTransformer, build_predict_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Representative dataset for INT8 calibration (200 windows from Part 1 train)
CALIBRATION_SAMPLES = 200


def load_model_checkpoint(model_class, checkpoint_path: Path, config: Config, device: str = "cpu"):
    """Load a model from checkpoint."""
    logger.info(f"Loading {model_class.__name__} from {checkpoint_path}")
    torch.load(checkpoint_path, map_location="cpu")

    # Determine which model class to use based on checkpoint structure
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Create model instance
    if model_class.__name__ == "VelocityModel":
        from aeronavis.models.velocity import VelocityModel, VelocityModelConfig

        VelocityModelConfig(
            window=config.model.window,
            stride=config.model.stride,
            hidden=config.model.velocity_hidden,
            max_params=config.model.max_params,
            tcn_channels=(32, 40, 64, 64, 64),
            tcn_kernel=5,
            tcn_dilations=(1, 2, 4, 8, 16),
            lstm_layers=2,
            lstm_hidden=64,
            fusion_hidden=128,
            dropout=0.1,
            residual_vel=0.2,
        )
        model = VelocityModel(
            VelocityModelConfig(
                window=200,
                stride=10,
                hidden=64,
                max_params=500000,
                tcn_channels=(32, 40, 64, 64, 64),
                tcn_kernel=5,
                tcn_dilations=(1, 2, 4, 8, 16),
                lstm_layers=2,
                lstm_hidden=64,
                fusion_hidden=128,
                dropout=0.1,
                residual_vel=0.2,
            )
        )
    elif model_class.__name__ == "PseudoOdoModel":
        model = PseudoOdoModel(
            PseudoOdoConfig(
                window=200,
                hidden=96,
                max_params=300000,
                cnn_channels=(48, 64, 96),
                cnn_kernel=5,
                gru_layers=2,
                gru_hidden=96,
                prenet_hidden=32,
                spec_bins=8,
                prenet_out=32,
                head_hidden=128,
                dropout=0.1,
            )
        )
    elif model_class.__name__ == "SlipClassifier":
        from aeronavis.models.slip import SlipClassifier, SlipConfig

        model = SlipClassifier(
            SlipConfig(
                window=200,
                hidden=64,
                max_params=200000,
                num_classes=3,
                dropout=0.1,
                share_trunk=True,
            )
        )
    elif model_class.__name__ == "VisualOdoNet":
        model = VisualOdoNet(
            VisualOdoConfig(
                input_size=224,
                in_channels=2,
                max_params=1_000_000,
                stem_channels=32,
                block_channels=(32, 32, 64, 64),
                fc_hidden=64,
                dropout=0.1,
            )
        )
    elif model_class.__name__ == "TextureGate":
        from aeronavis.models.texture_gate import TextureGate

        model = TextureGate(
            TextureGateConfig(
                input_size=96, num_classes=3, max_params=100_000, channels=(16, 32, 64), dropout=0.1
            )
        )
    elif model_class.__name__ == "PredictTransformer":
        from aeronavis.predict.model import PredictTransformer

        model = PredictTransformer(
            PredictTransformerConfig(
                input_dim=104,
                d_model=128,
                n_heads=4,
                n_layers=3,
                ff_dim=512,
                dropout=0.1,
                horizons=[5, 10, 15],
                max_len=30,
                max_params=2_000_000,
            )
        )
    else:
        raise ValueError(f"Unknown model class: {model_class}")

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    logger.info(
        f"Loaded {model_class.__name__} with {sum(p.numel() for p in model.parameters()):,} params"
    )
    return model


def get_calibration_dataset(config: Config, num_samples: int = CALIBRATION_SAMPLES) -> List[Dict]:
    """Load calibration dataset from Part 1 processed data."""
    # For now, generate synthetic calibration data
    # In production, load from data/processed/train/
    samples = []
    get_config()
    for _ in range(num_samples):
        # Random IMU window (6 channels: acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z)
        acc = np.random.randn(200, 3).astype(np.float32) * 2 + np.array([0, 0, 9.81])
        gyr = np.random.randn(200, 3).astype(np.float32) * 0.1
        imu = np.concatenate([acc, gyr], axis=1)  # (200, 6)

        # Random initial attitude
        np.eye(3).astype(np.float32)

        # Pseudo-odo inputs
        np.array([0.0], dtype=np.float32)

        # Visual inputs (for visual odo)
        img_t = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        img_t1 = np.random.randint(0, 255, (224, 224), dtype=np.uint8)

        # Texture gate input (96x96 grayscale)
        texture_img = np.random.randint(0, 255, (96, 96), dtype=np.uint8)

        # Predictor input (30 steps x 104 features)
        pred_input = np.random.randn(30, 104).astype(np.float32)

        samples.append(
            {
                "imu": imu,
                "att0": np.eye(3).astype(np.float32),
                "baro_delta": np.array([0.0], dtype=np.float32),
                "img_t": img_t,
                "img_t1": img_t1,
                "texture_img": texture_img,
                "predict_input": pred_input,
            }
        )
    return samples


def export_model_to_tflite(
    model: torch.nn.Module,
    sample_input: Dict,
    output_path: Path,
    quantize: bool = True,
    representative_dataset: Optional[List] = None,
) -> Dict:
    """Export a PyTorch model to TFLite format."""

    # Create example input tensors
    example_inputs = {}
    for key, value in sample_input.items():
        if isinstance(value, np.ndarray):
            example_inputs[key] = torch.from_numpy(value).unsqueeze(0).float()
        elif isinstance(value, np.generic):
            example_inputs[key] = torch.tensor(value).unsqueeze(0).float()

    # Export to ONNX first
    onnx_path = output_path.with_suffix(".onnx")
    torch.onnx.export(
        model,
        tuple(example_inputs.values()),
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=list(example_inputs.keys()),
        output_names=["output"],
        dynamic_axes={k: {0: "batch"} for k in example_inputs.keys()},
    )

    # Convert ONNX to TFLite using tf2onnx or onnx2tf
    # For simplicity, we'll use a subprocess call to onnx2tf
    import subprocess

    tflite_path = output_path.with_suffix(".tflite")

    cmd = [
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(output_path.parent),
        "-ois",
        "output",
        "--output_signature_defs",
        "True",
    ]

    if quantize:
        cmd.extend(["-qt", "int8"])
        if representative_dataset:
            cmd.extend(["-rsd", str(Path(representative_dataset).absolute())])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"onnx2tf failed: {result.stderr}")
        raise RuntimeError(f"TFLite conversion failed: {result.stderr}")

    # Verify TFLite model
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get model size
    model_size_mb = output_path.stat().st_size / (1024 * 1024)

    logger.info(f"Exported {output_path.name}: {model_size_mb:.2f} MB")
    logger.info(f"  Inputs: {len(input_details)}, Outputs: {len(output_details)}")
    for i, d in enumerate(input_details):
        logger.info(f"  Input {i}: {d['name']} {d['shape']} {d['dtype']}")
    for o, d in enumerate(output_details):
        logger.info(f"  Output {o}: {d['name']} {d['shape']} {d['dtype']}")

    return {
        "model_path": str(tflite_path),
        "onnx_path": str(onnx_path),
        "size_mb": model_size_mb,
        "input_details": [
            {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
            for d in input_details
        ],
        "output_details": [
            {"name": d["name"], "shape": d["shape"].tolist(), "dtype": str(d["dtype"])}
            for d in output_details
        ],
    }


def verify_tflite_parity(
    pytorch_model: torch.nn.Module,
    tflite_path: Path,
    test_samples: List[Dict],
    max_abs_error: float = 0.1,
) -> Dict:
    """Verify numerical parity between PyTorch and TFLite models."""
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    errors = []
    for sample in test_samples[:100]:  # Test on 100 samples
        # Prepare inputs
        for key, value in sample.items():
            if isinstance(value, np.ndarray):
                torch.from_numpy(value).unsqueeze(0).float()
                tf.convert_to_tensor(value[np.newaxis, ...].astype(np.float32))
            elif isinstance(value, np.generic):
                torch.tensor(value).unsqueeze(0).float()
                tf.convert_to_tensor(value[np.newaxis].astype(np.float32))

            # PyTorch forward
            with torch.no_grad():
                pytorch_out = pytorch_model(
                    **{k: v for k, v in sample.items() if isinstance(v, np.ndarray)}
                )

            # TFLite forward
            interpreter = tf.lite.Interpreter(model_path=str(model_path))
            interpreter.allocate_tensors()
            for i, detail in enumerate(input_details):
                interpreter.set_tensor(detail["index"], sample[key].astype(np.float32))
            interpreter.invoke()
            tflite_out = interpreter.get_tensor(output_details[0]["index"])

            # Compare
            if isinstance(pytorch_out, tuple):
                pytorch_out = pytorch_out[0]
            pytorch_np = (
                pytorch_out.numpy() if isinstance(pytorch_out, torch.Tensor) else pytorch_out
            )
            error = np.max(np.abs(pytorch_np - tflite_out))
            errors.append(error)
            if error > max_abs_error:
                logger.warning(f"Parity violation: error={error:.6f} > {max_abs_error}")

    return {
        "max_error": max(errors) if errors else 0,
        "mean_error": np.mean(errors) if errors else 0,
        "samples_tested": len(errors),
        "parity_passed": all(e <= max_abs_error for e in errors),
    }


def main():
    parser = argparse.ArgumentParser(description="Export NAV-X models to TFLite")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/tflite"), help="Output directory"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["velocity", "pseudo_odo", "slip", "visual_odo", "texture_gate", "forecaster"],
        help="Models to export",
    )
    parser.add_argument(
        "--quantize", action="store_true", default=True, help="Use INT8 quantization"
    )
    parser.add_argument("--verify", action="store_true", default=True, help="Verify TFLite parity")
    parser.add_argument(
        "--calibration-samples", type=int, default=200, help="Calibration samples for INT8"
    )
    args = parser.parse_args()

    get_config()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "assets").mkdir(parents=True, exist_ok=True)

    # Load calibration dataset
    calib_samples = get_calibration_dataset(get_config(), args.calibration_samples)

    # Model configurations
    model_builders = {
        "velocity": (VelocityModel, "models/velocity/best.pt"),
        "pseudo_odo": (PseudoOdoModel, "models/pseudo_odo/best.pt"),
        "slip": (SlipClassifier, "models/slip/best.pt"),
        "visual_odo": (VisualOdoNet, None),  # No checkpoint yet
        "texture_gate": (TextureGate, "models/texture_gate/best.pt"),
        "forecaster": (PredictTransformer, None),  # No checkpoint yet
    }

    manifest = {
        "models": {},
        "export_config": {
            "quantized": args.quantize,
            "calibration_samples": args.calibration_samples,
            "pytorch_version": torch.__version__,
        },
    }

    for model_name in args.models:
        if model_name not in model_builders:
            logger.warning(f"Unknown model: {model_name}")
            continue

        model_class, ckpt_path = model_builders[model_name]

        if ckpt_path and not Path(ckpt_path).exists():
            logger.warning(f"Checkpoint not found for {model_name}: {ckpt_path}, skipping")
            continue

        logger.info(f"Exporting {model_name}...")

        try:
            # Load model
            if ckpt_path:
                model = load_model_checkpoint(model_class, Path(ckpt_path), device="cpu")
            else:
                # Build from config
                if model_name == "velocity":
                    model = build_velocity_model(get_config())
                elif model_name == "pseudo_odo":
                    model = build_pseudo_odo_model(get_config())
                elif model_name == "slip":
                    model = build_slip_model(get_config())
                elif model_name == "visual_odo":
                    model = build_visual_odo(get_config())
                elif model_name == "texture_gate":
                    model = build_texture_gate(get_config())
                elif model_name == "forecaster":
                    model = build_predict_model(get_config())
                else:
                    raise ValueError(f"Unknown model: {model_name}")

            # Prepare sample input for export
            next(iter(calib_samples)) if calib_samples else {}

            # Export
            output_path = args.output_dir / "assets" / f"{model_name}.tflite"
            export_result = export_model_to_tflite(
                model=model,
                sample_input={},  # Will be filled by export function
                output_path=args.output_dir / "assets" / f"{model_name}.tflite",
                quantize=args.quantize,
                representative_dataset=calib_samples if args.quantize else None,
            )

            # Verify parity
            if args.verify:
                parity = verify_tflite_parity(
                    pytorch_model=model,
                    tflite_path=output_path,
                    test_samples=calib_samples[:100],
                )
                export_result["parity"] = parity

            manifest["models"][model_name] = export_result
            logger.info(f"✓ {model_name} exported successfully")

        except Exception as e:
            logger.error(f"Failed to export {model_name}: {e}")
            manifest["models"][model_name] = {"error": str(e)}

    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Manifest saved to {manifest_path}")

    # Create Android assets directory structure
    (args.output_dir / "assets").mkdir(exist_ok=True)

    logger.info("Export complete!")


if __name__ == "__main__":
    main()
