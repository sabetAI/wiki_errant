from contextlib import ExitStack
import errant
from tqdm import tqdm

from multiprocessing import Pool

def split(lst, n):
    """Yield successive n-sized chunks from lst."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

def merge(lst2):
    flat = []
    for l in lst2:
        flat += l
    return flat

# Process each line of all input files
def label_edits(pairs, args):
    annotator = errant.load("en")
    labels = []
    # Process each line of all input files
    for orig, cors in tqdm(pairs):
        label = []
        # Get the original and all the corrected texts
        orig = orig.strip()
        cors = [cors]
        # Skip the line if orig is empty
        if not orig: continue
        # Parse orig with spacy
        orig = annotator.parse(orig, args.tok)
        # Write orig to the output m2 file
        # Loop through the corrected texts
        for cor_id, cor in enumerate(cors):
            cor = cor.strip()
            # If the texts are the same, write a noop edit
            if orig.text.strip() == cor:
                label.append(noop_edit(cor_id).split('|||')[1])
            # Otherwise, do extra processing
            else:
                # Parse cor with spacy
                cor = annotator.parse(cor, args.tok)
                # Align the texts and extract and classify the edits
                edits = annotator.annotate(orig, cor, args.lev, args.merge)
                # Loop through the edits
                for edit in edits:
                    # Write the edit to the output m2 file
                    label.append(edit.to_m2(cor_id).split('|||')[1])
        # Write a newline when we have processed all corrections for each line
        labels.append(label)
    return labels

