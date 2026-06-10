from args import build_parser


def build_clustering_parser():
    parser = build_parser()
    parser.description = (
        "Run only phase 0 warm-up and server-side clustering for comparing "
        "dataset/factor combinations."
    )
    return parser
