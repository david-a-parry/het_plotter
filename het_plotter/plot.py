import bisect
import os
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')  # for headless servers
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from het_plotter.het_counter import counts_to_df


def add_highlight_to_plot(plt, region, color='black', pivot_table=None,
                          label_heatmaps=False):
    start = region[0]
    end = region[1]
    label = None
    if len(region) > 2:
        label = region[2]
    if pivot_table is not None:
        if not label_heatmaps:
            label = None
        h = len(pivot_table.axes[0])
        fpos = pivot_table.iloc[0]
        left = bisect.bisect_left(fpos.keys(), start)
        right = bisect.bisect_right(fpos.keys(), end)
        plt.plot([left, left], [0, h], '--', color=color, label=label)
        plt.plot([right, right], [0, h], '--', color=color)
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


def plot_zygosity(counter, contig, out_dir, logger, df=None, window_length=1e5,
                  plot_per_sample=True, plot_together=True, centromeres=dict(),
                  roi=dict()):
    if df is not None:
        frac_df = df
    else:
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
            ax = sns.heatmap(het_pivot)
            add_highlights(ax, contig, centromeres, roi, pivot=het_pivot)
            ax.set_title("Heterozygosity")
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
            ax.set_ylim(0, 1.1)
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
        fig = plt.figure(figsize=(18, 8))
        suptitle = fig.suptitle("{}".format(contig))
        plt.subplot(2, 1, 1)
        ax = sns.heatmap(frac_df.pivot("sample_id", "pos", "calls"),
                         cmap="viridis")
        ax.yaxis.set_label("")
        ax.xaxis.set_ticklabels([])
        ax.set_ylabel("Sample")
        ax.xaxis.set_visible(False)
        ax.set_title("Calls per Window")
        plt.yticks(rotation=0)
        plt.subplot(2, 1, 2)
        het_pivot = frac_df.pivot("sample_id", "pos", "het")
        ax = sns.heatmap(het_pivot)
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
        plt.subplots_adjust(hspace=0.4)
        plt.yticks(rotation=0)
        png = os.path.join(out_dir, "{}_{}.png".format(contig, s))
        png = os.path.join(out_dir, "{}_combined.png".format(contig))
        logger.info("Saving plots to {}".format(png))
        fig.savefig(png, bbox_extra_artists=(suptitle, lgd),
                    bbox_inches='tight')
        plt.cla()
        plt.close('all')
