import argparse
from contextlib import ExitStack
import errant

from tqdm import tqdm
import pandas as pd

from multiprocessing import Pool
from functools import partial

from utils import split, merge, label_edits

def main():
    # Parse command line args
    args = parse_args()
    print("Loading resources...")
    # Load Errant
    annotator = errant.load("en")
    # Open output m2 file
    out_m2 = open(args.out, "w")

    print("Processing parallel files...")
    # Process an arbitrary number of files line by line simultaneously. Python 3.3+
    # See https://tinyurl.com/y4cj4gth
    with ExitStack() as stack:
        orig_lines = stack.enter_context(open(args.orig, encoding='utf-8')).readlines()
        cor_lines = stack.enter_context(open(args.cor[0], encoding='utf-8')).readlines()
        pairs = list(zip(orig_lines, cor_lines))
        batch_size = len(orig_lines) // args.n_procs
        splits = split(pairs, batch_size)
        partial_func = partial(label_edits, args=args)

        with Pool(args.n_procs) as pool:
            results = pool.map(partial_func, splits)
        labeled = merge(results)

        for label in tqdm(labeled):
            out_m2.write(','.join(label) + '\n')
            
#    pr.disable()
#    pr.print_stats(sort="time")

# Parse command line args
def parse_args():
    parser=argparse.ArgumentParser(
        description="Align parallel text files and extract and classify the edits.\n",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [-h] [options] -orig ORIG -cor COR [COR ...] -out OUT")
    parser.add_argument(
        "-orig",
        help="The path to the original text file.",
        required=True)
    parser.add_argument(
        "-cor",
        help="The paths to >= 1 corrected text files.",
        nargs="+",
        default=[],
        required=True)
    parser.add_argument(
        "-out", 
        help="The output filepath.",
        required=True)
    parser.add_argument(
        "-n_procs", 
        type=int,
        help="N procs for multiprocessing",
        default = 60) 
    parser.add_argument(
        "-tok", 
        help="Word tokenise the text using spacy (default: False).",
        action="store_true")
    parser.add_argument(
        "-lev",
        help="Align using standard Levenshtein (default: False).",
        action="store_true")
    parser.add_argument(
        "-merge",
        help="Choose a merging strategy for automatic alignment.\n"
            "rules: Use a rule-based merging strategy (default)\n"
            "all-split: Merge nothing: MSSDI -> M, S, S, D, I\n"
            "all-merge: Merge adjacent non-matches: MSSDI -> M, SSDI\n"
            "all-equal: Merge adjacent same-type non-matches: MSSDI -> M, SS, D, I",
        choices=["rules", "all-split", "all-merge", "all-equal"],
        default="rules")
    args=parser.parse_args()
    return args

# Input: A coder id
# Output: A noop edit; i.e. text contains no edits
def noop_edit(id=0):
    return "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||"+str(id)

if __name__ == "__main__":
    main()
