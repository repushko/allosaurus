import argparse
import logging
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from allosaurus.app import read_recognizer
from allosaurus.constants import EMBEDDING_FILE_SUFFIX, AGGREGATED_EMBEDDINGS_FILE_NAME


def parse_args() -> argparse.Namespace:
    arguments_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    arguments_parser.add_argument(
        "-src",
        "--src_dir",
        help="Source directory (where .wav files stored) for phoneme embedding computation.",
        type=Path,
        required=True,
    )
    arguments_parser.add_argument(
        "-m",
        "--model_name",
        help="Name of allosaurus model to use.",
        type=str,
        required=False,
        default="interspeech21",
    )
    arguments_parser.add_argument(
        "-out",
        "--out_filename",
        help="Filename for output phoneme-to-embedding mapping.",
        type=Path,
        required=False,
        default=Path() / "phoneme_to_embedding_mapping.pkl"
    )
    return arguments_parser.parse_args()


def main(src_dir: Path, model_name: str, out_filename: Path) -> None:
    # Setup logger
    logging.basicConfig(level=logging.getLevelName("INFO"))
    logger = logging.getLogger(Path(__file__).name)

    # Collect audios to compute embeddings
    audios = list(src_dir.rglob("*.wav"))
    logger.info(f"Num. of audios to compute embeddings: {len(audios)}")

    logger.info(f"Initializing Allosaurus {model_name} model...")
    model = read_recognizer(model_name, return_embeddings=True)
    logger.info(f"Allosaurus model {model_name} successfully initialized.")

    phoneme_to_embedding = defaultdict(list)
    for audio in tqdm(audios, desc="Embeddings extraction..."):
        phonemes, embeddings = model.recognize(str(audio))
        phonemes = phonemes.split(" ")

        with audio.with_suffix(EMBEDDING_FILE_SUFFIX).open("wb") as file:
            pickle.dump((phonemes, embeddings), file)

        for phoneme, embedding in zip(phonemes, embeddings):
            phoneme_to_embedding[phoneme].append(embedding)
    logger.info("All embeddings have been successfully extracted.")

    raw_mapping_save_path = src_dir / AGGREGATED_EMBEDDINGS_FILE_NAME
    with raw_mapping_save_path.open("wb") as file:
        pickle.dump(phoneme_to_embedding, file)
    logger.info(f"Saved raw phoneme-to-embedding map to {raw_mapping_save_path}.")

    phoneme_to_embedding = {k: np.mean(v, axis=0) for k, v in phoneme_to_embedding.items()}
    logger.info("Phoneme-to-embedding map successfully computed.")

    with out_filename.open("wb") as file:
        pickle.dump(phoneme_to_embedding, file)
    logger.info(f"Phoneme-to-embedding map successfully saved to {out_filename}")


def run():
    args = parse_args()
    main(**vars(args))


if __name__ == '__main__':
    run()