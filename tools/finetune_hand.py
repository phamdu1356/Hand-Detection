"""Compare a short hand-pose fine-tune with the current weights on the same held-out images."""
import argparse
import json
import random
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="hand-keypoints dataset directory")
    parser.add_argument("--weights", default="hand_pose.pt")
    parser.add_argument("--output", type=Path, required=True, help="new experiment directory")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    source = args.data.resolve()
    output = args.output.resolve()
    rng = random.Random(42)
    train = sorted((source / "images/train").glob("*.jpg"))
    val = sorted((source / "images/val").glob("*.jpg"))
    if len(train) < 2048 or len(val) < 768:
        parser.error("need 2048 training and 768 validation images")
    output.mkdir(parents=True, exist_ok=False)
    rng.shuffle(train)
    rng.shuffle(val)
    for name, paths in [("train", train[:2048]), ("val", val[:256]), ("test", val[256:768])]:
        (output / f"{name}.txt").write_text("".join(f"{p}\n" for p in paths))
    spec = yaml.safe_load((source / "data.yaml").read_text())
    dataset = output / "data.yaml"
    dataset.write_text(yaml.safe_dump(dict(path=str(output), names=spec["names"],
                                          kpt_shape=spec["kpt_shape"], train="train.txt",
                                          val="val.txt", test="test.txt")))
    opts = dict(data=str(dataset), imgsz=640, device=args.device, workers=2,
                batch=8, project=str(output), plots=False, verbose=False)

    def score(weights, name):
        result = YOLO(str(weights)).val(**opts, split="test", name=name)
        return {"box_map50_95": float(result.box.map), "pose_map50_95": float(result.pose.map)}

    model = YOLO(args.weights)
    model.train(**opts, name="fit", epochs=args.epochs, optimizer="AdamW", lr0=0.00015,
                seed=42, amp=False, fliplr=0, mosaic=0.25, close_mosaic=0, freeze=10)
    baseline = score(args.weights, "baseline")
    candidate = score(model.trainer.best, "candidate")
    result = {"train_images": 2048, "validation_images": 256, "evaluation_images": 512,
              "seed": 42, "baseline": baseline, "candidate": candidate,
              "candidate_weights": str(model.trainer.best)}
    print(json.dumps(result, indent=2))
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
