import numpy as np
import pandas as pd
from collections import defaultdict

gt_ids = ['het', 'hom_alt', 'hom_ref']
gt_tups = [(0, 1), (1, 1), (0, 0)]


def counts_to_df(counter, window_length):
    fract_dict = defaultdict(list)
    positions = np.array(counter.pos)
    count_array = np.array(counter.counts)
    i = 0
    while i < len(positions):
        window = np.where(np.logical_and(positions >= positions[i],
                                         positions <= (positions[i] +
                                                       window_length)))
        if not window:
            i += 1
            continue
        c = count_array[window]
        f = np.divide(np.sum(c, axis=0), len(c))
        for s in counter.samples:
            fract_dict['pos'].append(np.median(positions[i]))
            fract_dict['sample_id'].append(s)
            fract_dict['het'].append(
                f[counter.gt_indices['het']][counter.samp_indices[s]])
            fract_dict['calls'].append(
                np.sum(np.sum(c, axis=0), axis=0)[counter.samp_indices[s]])
        i = window[-1][-1] + 1
    return pd.DataFrame(fract_dict)


class PosCounter(object):
    ''' Count number of hets/homs per contig for a sample.'''

    def __init__(self, samples):
        '''
           For a given set of samples create PosCounter object that will
           count genotype calls per position (for a single contig).

           Counts are recorded in a 2d matrix for each position and
           stored as entries to in the 'count' attribute, pointing to a
           list of these counts. Positions are recorded as a list in the
           order encountered in the 'pos' attribute.
        '''
        self.samp_indices = dict((s, n) for n, s in enumerate(samples))
        self.gt_indices = dict((r, n) for n, r in enumerate(gt_ids))
        self.gt_indices.update(dict((r, n) for n, r in enumerate(gt_tups)))
        self.gt_indices[(1, 0)] = 0  # in case genotypes are phased
        self.counts = []
        self.pos = []
        self.samples = samples

    def count_genotype(self, pos, samples, gts):
        '''
            Count genotypes for all samples at given position. Assumes
            the same position will only be processed once. If samples
            and gts are empty nothing will be done.

            Args:
                pos:    coordinate of variant. Appended to self.pos
                        attribute which is in the same order as
                        self.counts.

                samples:
                        list of sample IDs

                gts:    list of genotypes for each sample, in the same
                        order as provided samples. Valid values are
                        in string or tuple form are 'het' or (0, 1),
                        'hom_alt' or (1, 1) and 'hom_ref' or (0, 0).
        '''
        if len(samples) != len(gts):
            raise ValueError("samples and gts must be the same length.")
        if not samples:
            return
        c = np.zeros((len(gt_ids), len(self.samples)), dtype=int)
        for i in range(len(samples)):
            c[self.gt_indices[gts[i]], self.samp_indices[samples[i]]] += 1
        self.pos.append(pos)
        self.counts.append(c)
