import argparse
from collections import Counter
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from src.common.config import load_yaml
from src.common.paths import ROOT, RUNS_DIR
from src.inference.pipeline import Detection, HummingbirdPipeline

GREEN, ORANGE = (0, 200, 0), (0, 165, 255)
WINDOW = "beija-flor"


def draw(frame, detection: Detection, text_scale: float) -> None:
    x1, y1, x2, y2 = detection.box
    name = detection.species.replace("_", " ")
    prefix = f"#{detection.track_id} " if detection.track_id is not None else ""
    if detection.uncertain:
        color, text = ORANGE, f"{prefix}Incerta ({name} {detection.confidence:.0%})"
    else:
        color, text = GREEN, f"{prefix}{name} {detection.confidence:.0%}"

    scale = text_scale * max(frame.shape[0] / 700, 0.8)
    thickness = max(2, round(scale * 2))
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    top = max(y1 - h - baseline, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.rectangle(frame, (x1, top), (x1 + w, top + h + baseline), color, -1)
    cv2.putText(frame, text, (x1, top + h), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detecta beija-flores e identifica a espécie em um vídeo")
    parser.add_argument("source", type=Path, help="caminho do vídeo")
    parser.add_argument("--name", help="nome da corrida (padrão: run_name do inference.yaml)")
    parser.add_argument("--max-frames", type=int, help="processa só os primeiros N quadros")
    parser.add_argument("--no-show", action="store_true", help="não abre a janela do vídeo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml("inference")
    name = args.name or cfg["run_name"]
    max_frames = args.max_frames or cfg["max_frames"]
    show = cfg["show_window"] and not args.no_show

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise FileNotFoundError(f"Não foi possível abrir {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    out_path = RUNS_DIR / "pipeline" / name / f"{args.source.stem}_species.mp4"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    if show:
        window_width = min(cfg["window_max_width"], width)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, window_width, round(window_width * height / width))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = HummingbirdPipeline(cfg, device)

    n_frames = detected_frames = uncertain = 0
    species_counts = Counter()

    try:
        for _ in tqdm(range(total)):
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1

            detections = pipeline.process(frame)
            detected_frames += bool(detections)
            for detection in detections:
                draw(frame, detection, cfg["text_scale"])
                if detection.uncertain:
                    uncertain += 1
                else:
                    species_counts[detection.species.replace("_", " ")] += 1

            writer.write(frame)
            if show:
                cv2.imshow(WINDOW, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    print(f"Quadros: {n_frames} | com detecção: {detected_frames} ({detected_frames / max(n_frames, 1):.0%})")
    print(f"Aves acompanhadas (IDs): {pipeline.n_tracks}")
    print(f"Previsões com confiança >= {cfg['min_species_conf']:.0%}: {sum(species_counts.values())} | incertas: {uncertain}")
    for species, n in species_counts.most_common():
        print(f"  {species}: {n}")
    print(f"Vídeo salvo em: {out_path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()