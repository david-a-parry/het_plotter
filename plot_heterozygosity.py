#!/usr/bin/env python3
import sys
import os
import re
import argparse
import bisect
import gzip
import logging
import pysam
import tempfile
import scipy.stats
import multiprocessing as mp
import numpy as np
from shutil import copyfile,copyfileobj
from itertools import repeat
from collections import defaultdict
from copy import copy
from parse_vcf import VcfReader
from vase.ped_file import PedFile
import matplotlib
if os.environ.get('DISPLAY','') == '':
    matplotlib.use('Agg') #for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns

chrom_re = re.compile(r'''^(chr)?[1-9][0-9]?$''')
gt_ids = ['het', 'hom_alt', 'hom_ref',]
gt_tups = [(0, 1), (1, 1), (0, 0)]
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
        self.samp_indices = dict((s, n) for n,s in enumerate(samples))
        self.gt_indices = dict((r, n) for n,r in enumerate(gt_ids))
        self.gt_indices.update(dict((r, n) for n,r in enumerate(gt_tups)))
        self.gt_indices[(1, 0)] = 0 #in case genotypes are phased
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

def initialize_mp_logger(logger, loglevel, logfile=None):
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
           '[%(asctime)s] HetPlotter-%(processName)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

def _process_runner(tup):
    kwargs1, kwargs2 = tup
    kwargs2.update(kwargs1)
    return get_gt_counts(**kwargs2)

def get_logger(loglevel=logging.INFO, logfile=None):
    logger = logging.getLogger("HetPlotter")
    logger.setLevel(loglevel)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
           '[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if logfile is not None:
        fh = logging.FileHandler(logfile)
        fh.setLevel(logger.level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def counts_to_df(counter, window_length):
    fract_dict = defaultdict(list)
    positions = np.array(counter.pos)
    count_array = np.array(counter.counts)
    i = 0
    while i < len(positions):
        #print("At position {}".format(positions[i]))
        window = np.where(np.logical_and(positions>=positions[i],
									positions<=(positions[i] + window_length)))
        if not window:
            i += 1
            continue
        c = count_array[window]
        p = positions[window]
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

def add_highlight_to_plot(plt, region, color='black', pivot_table=None,
                          label_heatmaps=False):
    start = region[0]
    end = region[1]
    label = None
    if len(region) > 2:
        label = region[2]
    if pivot_table is not None:
        if not label_heatmaps:
            label=None
        h = len(pivot_table.axes[0])
        fpos = pivot_table.iloc[0]
        l = bisect.bisect_left(fpos.keys(), start)
        r = bisect.bisect_right(fpos.keys(), end)
        plt.plot([l, l], [0, h], '--', color=color, label=label)
        plt.plot([r, r], [0, h], '--', color=color)
    else:
        plt.plot([start, start], [0, 1], '--',
                 color=color, label=label)
        plt.plot([end, end], [0, 1], '--', color=color)

def add_highlights(plt, contig, centromeres, roi, pivot=None,
                   label_heatmaps=False):
    if contig in centromeres:
        if pivot is None:
            col = 'black'
        else:
            col = 'lightgray'
        add_highlight_to_plot(plt, centromeres[contig], color=col,
                              pivot_table=pivot, label_heatmaps=label_heatmaps)
    if contig in roi and len(roi[contig]) > 0:
        pal = sns.color_palette("Set2", len(roi[contig]))
        for region, col in zip(roi[contig], pal):
            add_highlight_to_plot(plt, region, color=col, pivot_table=pivot,
                                  label_heatmaps=label_heatmaps)

def plot_zygosity(counter, contig, out_dir, logger, window_length=1e5,
                  plot_per_sample=True, plot_together=True, centromeres=dict(),
                  roi=dict()):
    logger.info("Converting counts for chromosome {} to ".format(contig) +
                "dataframe")
    frac_df = counts_to_df(counter, window_length)
    if plot_per_sample:
        for s in counter.samples:
            grid = plt.GridSpec(2, 2, wspace=0.1, hspace=0.2)
            samp_df = frac_df[frac_df.sample_id == s]
            fig = plt.figure(figsize=(18,8))
            suptitle = fig.suptitle("{} {}".format(s, contig))
            fig.add_subplot(grid[0, 0],)
            ax = sns.heatmap(samp_df.pivot("sample_id", "pos", "calls"),
                                           cmap="viridis")
            ax.set_title("Calls per Window")
            ax.xaxis.set_ticklabels([])
            ax.set_ylabel("")
            fig.add_subplot(grid[1, 0],)
            het_pivot = samp_df.pivot("sample_id", "pos", "het")
            ax = sns.heatmap(het_pivot)#, cmap="plasma")
            add_highlights(ax, contig, centromeres, roi, pivot=het_pivot)
            ax.set_title("Heterozygosity")
            #ax.xaxis.set_major_formatter(
            #    mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
            xticklabels = []
            for it in ax.get_xticklabels():
                if it.get_text():
                    it.set_text('{:,}'.format(int(float(it.get_text()))))
                    xticklabels += [it]
            if xticklabels:
                ax.set_xticklabels(xticklabels) 
            ax.set_xlabel("Position")
            ax.set_ylabel("")
            ax = fig.add_subplot(grid[0, 1])
            ax.set_title("Calls per Window")
            ax.plot(samp_df['pos'], samp_df['calls'], 'r', linestyle=':',
                    label="Calls per {:g} bp".format(window_length))
            ax.xaxis.set_ticklabels([])
            ax.set_ylabel("Calls")
            ax = fig.add_subplot(grid[1, 1])
            ax.plot(samp_df['pos'], samp_df['het'])
            add_highlights(ax, contig, centromeres, roi)
            ax.set_title("Heterozygosity")
            ax.set_ylabel("Heterozygosity")
            ax.set_xlabel("Position)")
            ax.set_ylim(0,1.1)
            ax.xaxis.set_major_formatter(
                mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
            plt.xticks(rotation=90)
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2,
                            borderaxespad=0.)
            png = os.path.join(out_dir, "{}_{}.png".format(contig, s))
            logger.info("Saving {}".format(png))
            fig.savefig(png, bbox_extra_artists=(suptitle, lgd),
                        bbox_inches='tight')
            plt.cla()
            plt.close('all')
    if plot_together:
        logger.info("Plotting all samples for chromosome {}".format(contig))
        fig = plt.figure(figsize=(18,8))
        suptitle = fig.suptitle("{}".format(contig))
        plt.subplot(2, 1, 1)
        ax = sns.heatmap(frac_df.pivot("sample_id", "pos", "calls"),
                         cmap="viridis")
        ax.yaxis.set_label("")
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel("Sample")
        #ax.set_xlabel("Position")
        ax.xaxis.set_visible(False)
        ax.set_title("Calls per Window")
        plt.yticks(rotation=0)
        plt.subplot(2, 1, 2)
        het_pivot = frac_df.pivot("sample_id", "pos", "het")
        ax = sns.heatmap(het_pivot)#, cmap="plasma")
        add_highlights(ax, contig, centromeres, roi, pivot=het_pivot,
                       label_heatmaps=True)
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, bbox_to_anchor=(1.25, 1), loc=2,
                        borderaxespad=0.)
        ax.set_title("Heterozygosity")
        ax.set_ylabel("Sample")
        ax.set_xlabel("Position")
        xticklabels = []
        for it in ax.get_xticklabels():
            if it.get_text():
                it.set_text('{:,}'.format(int(float(it.get_text()))))
                xticklabels += [it]
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        #ax.xaxis.set_major_formatter(
        #    mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
        plt.subplots_adjust(hspace=0.4)
        plt.yticks(rotation=0)
        png = os.path.join(out_dir, "{}_{}.png".format(contig, s))
        png = os.path.join(out_dir, "{}_combined.png".format(contig))
        logger.info("Saving plots to {}".format(png))
        fig.savefig(png, bbox_extra_artists=(suptitle, lgd),
                    bbox_inches='tight')
        plt.cla()
        plt.close('all')

def get_gt_counts(vcf, samples, contig, out_dir=None, coordinate_table=None,
                  parents=None, children=set(), prog_interval=10000,
                  window_length=1e5, roi=dict(), centromeres=dict(),
                  logger=None, loglevel=logging.INFO, plot=True):
    vreader = VcfReader(vcf)
    if logger is None:
        logger = mp.get_logger()
        if logger.level == logging.NOTSET:
            initialize_mp_logger(logger, loglevel)
    vreader.set_region(contig)
    logger.info("Reading chromosome {}".format(contig))
    gt_fields = ['GT', 'GQ', 'DP', 'AD']
    gt_counter = PosCounter(samples)
    n = 0
    valid = 0
    for record in vreader:
        if n % prog_interval == 0 and n != 0:
            logger.info("Read {:,} records, processed {:,},".format(n, valid) +
                        " {:,} filtered at {}:{}".format(n - valid,
                                                         record.CHROM,
                                                         record.POS))
        n += 1
        if len(record.ALLELES) != 2: #biallelic only
            continue
        if len(record.REF) != 1 or len(record.ALT) != 1: #SNVs only
            continue
        if record.FILTER != 'PASS':
            continue
        valid += 1
        gts = record.parsed_gts(fields=gt_fields, samples=samples)
        called_samps = []
        samp_gts = []
        for s in samples:
            #filter on GQ/DP
            if not gt_passes_filters(gts, s):
                continue
            called_samps.append(s)
            samp_gts.append(gts['GT'][s])
        gt_counter.count_genotype(pos=record.POS, samples=called_samps,
                                  gts=samp_gts)
    logger.info("Finished processing variants for contig {}".format(contig))
    if plot:
        plot_zygosity(gt_counter, contig, out_dir, logger,
                      window_length=window_length, centromeres=centromeres,
                      roi=roi)
    else:
        return gt_counter

def read_roi_bed(bedfile):
    roi = defaultdict(list)
    with open(bedfile, 'rt') as infile:
        for line in infile:
            cols = line.split()
            if len(cols) < 3:
                sys.exit("Not enough columns in bed for {}".format(bedfile))
            region = [int(cols[1]) + 1, int(cols[2])]
            if len(cols) > 3:
                region.append(cols[3])
            roi[cols[0]].append(region)
    return roi

def read_centromere_bed(bedfile):
    centromeres = dict()
    with open(bedfile, 'rt') as infile:
        for line in infile:
            cols = line.split()
            if len(cols) < 3:
                sys.exit("Not enough columns in bed for {}".format(bedfile))
            centromeres[cols[0]] = (int(cols[1]) + 1, int(cols[2]),
                                    'centromere')
    return centromeres

def main(vcf, samples=[], chromosomes=[], output_directory='zygosity_plots',
         build=None, roi=None, progress_interval=10000, threads=1,
         window_size=1e5, force=False):
    vreader = VcfReader(vcf)
    logger = get_logger()
    centromeres = dict()
    if build is not None:
        cfile = os.path.join(os.path.dirname(__file__), "data",
                             build + "_centromeres.bed")
        if os.path.exists(cfile):
            centromeres = read_centromere_bed(cfile)
        else:
            sys.exit("No centromere file ({}) for build {}".format(cfile,
                                                                   build))
    highlight_regions = dict()
    if roi is not None:
        highlight_regions = read_roi_bed(roi)
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    elif not force:
        sys.exit("Output directory '{}' exists ".format(output_directory) +
                 "- use --force to overwrite")
    if not samples:
        samples = vreader.header.samples
    if not chromosomes:
        chromosomes = get_seq_ids(vcf)
    kwargs = {'vcf': vcf, 'prog_interval': progress_interval,
              'out_dir': output_directory, 'samples': samples,
              'window_length': window_size, 'roi': highlight_regions,
              'centromeres': centromeres, 'plot': True}
    contig_args = ({'contig': x} for x in chromosomes)
    if not contig_args:
        raise RuntimeError("No valid contigs identified in {}".format(fai))
    if threads > 1:
        with mp.Pool(threads) as p:
            x = p.map(_process_runner, zip(contig_args, repeat(kwargs)))
            #x.get()
    else:
        for c in contig_args:
            kwargs.update(c)
            get_gt_counts(**kwargs)
    logger.info("Finished.")

def get_seq_ids(vcf):
    if vcf.endswith(".bcf"):
        idx = vcf + '.csi'
        preset = 'bcf'
    else:
        idx = vcf + '.tbi'
        preset = 'vcf'
    if not os.path.isfile(idx):   #create index if it doesn't exist
        pysam.tabix_index(vcf, preset=preset)
    if preset == 'bcf':
        vf = pysam.VariantFile(vcf)
        return (c for c in vf.index.keys() if chrom_re.match(c))
    else:
        tbx = pysam.TabixFile(vcf)
        return (c for c in tbx.contigs if chrom_re.match(c))

def gt_passes_filters(gts, s, min_gq=20, min_dp=10, min_ab=0.25):
    if gts['GQ'][s] is None or gts['GQ'][s] < min_gq:
        return False
    if gts['DP'][s] is None or gts['DP'][s] < min_dp:
        return False
    if gts['GT'][s] == (0, 1):
        #filter hets on AB
        dp = sum(gts['AD'][s])
        ad = gts['AD'][s][1]
        if ad is not None and dp > 0:
            ab = ad/dp
            if ab < min_ab:
                return False
    return True


def get_parser():
    parser = argparse.ArgumentParser(
                  description='Plot heterozygosity from a VCF.')
    parser.add_argument("-i", "--vcf", "--input", required=True,
                        help="Input VCF file.")
    #parser.add_argument("-p", "--ped", help='''PED file detailing any familial
    #                    relationships for samples in VCF.''')
    parser.add_argument("-s", "--samples", nargs='+', help='''One or more
                        samples to plot. Defaults to all in VCF.''')
    parser.add_argument("-c", "--chromosomes", nargs='+', help='''One or more
                        chromosomes to plot. Defaults to all in standard
                        chromosomes.''')
    parser.add_argument("-o", "--output_directory", default='zygosity_plots',
                        help='''Directory to place plot PNG files.
                                Default='zygosity_plots'.''')
    parser.add_argument("-w", "--window_size", type=float, default=1e5,
                        help='''Windows size to use when calculating ratios of
                        heterozygous vs homozygous genotypes. Default=1e5.''')
    parser.add_argument("-t", "--threads", type=int, default=1, help='''Number
                         of threads to use. Default=1. Maximum will be
                        determined by the number of chromosomes in your
                        reference.''')
    parser.add_argument("--progress_interval", type=int, default=10000,
                        metavar='N', help='''Report progress every N
                        variants.''')
    parser.add_argument("-b", "--build", help='''Genome build. If specified,
                        centromeres will be marked if there is a corresponding
                        BED file of centromere locations in the 'data'
                        subdirectory (hg19, hg38, GRCh37, GRCh38 available by
                        default).''')
    parser.add_argument("-r", "--roi", help='''BED file of regions of interest.
                        If a 4th column is present this will be used to label
                        these regions.''')
    parser.add_argument("--force", action='store_true', help='''Overwrite
                        existing output directories.''')
    return parser

if __name__ == '__main__':
    argparser = get_parser()
    args = argparser.parse_args()
    main(**vars(args))

