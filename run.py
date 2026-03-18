import argparse
from datetime import datetime

from utils.test_models import *
from utils.train_models import *
from utils.infer import infer_uniamp
from utils.functions import parse_arguments


def train(args):
    if args.model == "bert":
        train_bert(args)
    elif args.model == "uniamp":
        train_uniamp(args)
    else:
        raise ValueError("Model input error")
    pass


def test(args):
    test_uniamp(args)


def run(args):
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./models", exist_ok=True)
    os.makedirs("./checkpoint", exist_ok=True)

    if args.mode == "train":
        train(args)
    elif args.mode == "pretrain":
        if args.model != "bert":
            print(
                f'Mode is "pretrain", overriding model setting to "bert" (was "{args.model}").'
            )
            args.model = "bert"
        print(
            f'Mode is "pretrain", overriding feature setting to ["bert"] (was "{args.feature}").'
        )
        args.feature = ["bert"]

        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "infer":
        infer_uniamp(args)
    else:
        raise ValueError("Mode input error")


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
