from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import math
import os


def get_roc_auc_fig(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores[0].shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']
    method_name = ['CollaGAN', 'Syn-Seg', 'Cls-3P', 'DiagnosisGAN']
    font_size = 11

    auc = np.array([])
    opt_thr = np.array([])
    for i in range(num_classes):
        for idx, scores in enumerate(output_scores):
            fpr, tpr, thr = metrics.roc_curve(true_labels, scores[:, i], pos_label=i)
            auc = metrics.auc(fpr, tpr)
            best_idx = np.argmax(1 - fpr + tpr)
            opt_thr = np.append(opt_thr, thr[best_idx])

            if save_roc and not math.isnan(auc):
                plt.grid(linestyle='--')
                plt.axes().set_axisbelow(True)
                plt.axes().set_aspect('equal')
                plt.title(class_name[i], fontsize=font_size)
                plt.plot(fpr, tpr, label=method_name[idx] + ' (AUC=' + '{:.1f})'.format(auc*100), zorder=1)
                plt.xticks(fontsize=font_size)
                plt.yticks(fontsize=font_size)
                plt.xlabel('1 - Specificity', fontsize=font_size)
                plt.ylabel('Sensitivity', fontsize=font_size)

            if i == 2:
                loc = 4
            else:
                loc = 4

        plt.legend(loc=loc, fontsize=font_size)

        plt.savefig(os.path.join(figure_name, class_name[i] + '.svg'), bbox_inches='tight', format='svg')
        plt.cla()

        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))

    if find_opt_thr:
        return auc, opt_thr

    return auc


def get_roc_auc(true_labels, output_scores, find_opt_thr=False, save_roc=False, figure_name='roc.png'):
    num_classes = output_scores.shape[1]

    class_name = ['Oncocytoma', 'AML', 'chRCC', 'pRCC', 'ccRCC']

    if num_classes is 2:
        fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, 1])
        auc = metrics.auc(fpr, tpr)
        best_idx = np.argmax(1 - fpr + tpr)
        opt_thr = thr[best_idx]
        if save_roc:
            plt.plot(fpr, tpr, label=', AUC: ' + '{:.3f}'.format(auc))
    else:
        auc = np.array([])
        opt_thr = np.array([])
        for i in range(num_classes):
            fpr, tpr, thr = metrics.roc_curve(true_labels, output_scores[:, i], pos_label=i, drop_intermediate=False)
            auc = np.append(auc, metrics.auc(fpr, tpr))
            best_idx = np.argmax(1 - fpr + tpr)
            opt_thr = np.append(opt_thr, thr[best_idx])
            if save_roc and not math.isnan(auc[i]):
                # plt.plot(fpr * 100, tpr * 100, label='C: ' + str(i) + ', AUC: ' + '{:.1f}'.format(auc[i] * 100))
                plt.plot(fpr, tpr, label=class_name[i] + ': AUC: ' + '{:.1f}'.format(auc[i]))
        # auc = np.append(auc, metrics.roc_auc_score(true_labels, output_scores, multi_class='ovr'))
        auc = np.append(auc, np.nanmean(auc))
        if save_roc:
            plt.title('Internal DB (avg. AUC: ' + '{:.1f})'.format(auc[num_classes]))
    if save_roc:
        plt.xlabel('1 - specificity (%)')
        plt.ylabel('Sensitivity (%)')
        plt.legend()
        plt.savefig(figure_name)
        plt.cla()

    if find_opt_thr:
        return auc, opt_thr

    return auc


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name,
                layoutopts={'plotly': {'yaxis': {'range': [0, 1], 'autorange': True}}},
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name,
                          update='append')