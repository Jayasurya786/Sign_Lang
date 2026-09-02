import argparse
import sys
from pathlib import Path

if __package__ in {None, ''}:
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from src.data.landmarks import extract_hand_landmarks
from src.models.bilstm import train_bilstm_from_csv
from src.models.realtime import load_trained_model, predict_live
from src.processing.cleaning import save_cleaned_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Sign language recognition pipeline')
    subparsers = parser.add_subparsers(dest='command', required=True)

    landmarks_parser = subparsers.add_parser('landmarks', help='Extract hand landmark features from image samples')
    landmarks_parser.add_argument('--input', type=str, required=True)
    landmarks_parser.add_argument('--output', type=str, default='data/processed/landmarks.csv')

    clean_parser = subparsers.add_parser('clean', help='Clean and normalize landmark dataset')
    clean_parser.add_argument('--input', type=str, required=True)
    clean_parser.add_argument('--output', type=str, default='data/processed/cleaned.csv')

    train_parser = subparsers.add_parser('train-model', help='Train a BiLSTM on the processed landmark dataset')
    train_parser.add_argument('--input', type=str, required=True)
    train_parser.add_argument('--output', type=str, default='models/sign_bilstm.keras')
    train_parser.add_argument('--sequence-length', type=int, default=30)
    train_parser.add_argument('--epochs', type=int, default=20)
    train_parser.add_argument('--batch-size', type=int, default=32)

    predict_parser = subparsers.add_parser('predict-live', help='Run webcam prediction using a saved BiLSTM model')
    predict_parser.add_argument('--model', type=str, required=True)
    predict_parser.add_argument('--labels', type=str, nargs='+', required=True)
    predict_parser.add_argument('--camera-index', type=int, default=0)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == 'landmarks':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        extract_hand_landmarks(args.input, args.output)
        print(f"Saved landmark dataset to {args.output}")
        return

    if args.command == 'clean':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        cleaned = save_cleaned_dataset(args.input, args.output)
        print(f"Saved cleaned dataset to {args.output} with {len(cleaned)} rows.")
        return

    if args.command == 'train-model':
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        model, history, class_names = train_bilstm_from_csv(
            args.input,
            model_path=args.output,
            sequence_length=args.sequence_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
        print(f"Trained BiLSTM saved to {args.output} with classes: {class_names}")
        print(f"Final training accuracy: {history.history['accuracy'][-1] if 'accuracy' in history.history else 'n/a'}")
        return

    if args.command == 'predict-live':
        model = load_trained_model(args.model)
        predict_live(model, args.labels, camera_index=args.camera_index)
        return


if __name__ == '__main__':
    main()
