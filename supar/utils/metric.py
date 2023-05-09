# -*- coding: utf-8 -*-

from collections import Counter
import re

class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.


class ArgumentMetric(Metric):
    def __init__(self, eps=1e-12):
        super(Metric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0

        self.predicate_t = 0.0
        self.predicate_pred = 0.0
        self.predicate_gold = 0.0

        self.span_t = 0.0
        self.span_pred = 0.0
        self.span_gold = 0.0

        self.eps = eps

    def __call__(self, preds, golds, pred_p, gold_p, pred_span, gold_span):
        """
        preds, golds: [batch_size, seq_len, seq_len, seq_len]
        pred_p, gold_p: [batch_size, seq_len]
        pred_span, gold_span: [batch_size, seq_len, seq_len]
        """
        # TODO: add predicate and span metric
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()

        pred_p_mask = pred_p.gt(0)
        gold_p_mask = gold_p.gt(0)
        p_mask = pred_p_mask & gold_p_mask
        self.predicate_pred += pred_p_mask.sum().item()
        self.predicate_gold += gold_p_mask.sum().item()
        self.predicate_t += p_mask.sum().item()

        pred_s_mask = pred_span.gt(0)
        gold_s_mask = gold_span.gt(0)
        s_mask = pred_s_mask & gold_s_mask
        self.span_pred += pred_s_mask.sum().item()
        self.span_gold += gold_s_mask.sum().item()
        self.span_t += s_mask.sum().item()

        return self

    def __repr__(self):
        return f"P_P: {self.p_p:6.2%} P_R: {self.p_r:6.2%} P_F: {self.p_f:6.2%}  S_P: {self.s_p:6.2%} S_R:{self.s_r:6.2%} S_F: {self.s_f:6.2%} UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def p_p(self):
        return self.predicate_t / (self.predicate_pred + self.eps)
    
    @property
    def p_r(self):
        return self.predicate_t / (self.predicate_gold + self.eps)

    @property
    def p_f(self):
        return 2 * self.predicate_t / (self.predicate_pred + self.predicate_gold + self.eps)

    @property
    def s_p(self):
        return self.span_t / (self.span_pred + self.eps)

    @property
    def s_r(self):
        return self.span_t / (self.span_gold + self.eps)

    @property
    def s_f(self):
        return 2 * self.span_t / (self.span_pred + self.span_gold + self.eps)

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)


class AttachmentMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.eps = eps

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.total = 0.0
        self.correct_arcs = 0.0
        self.correct_rels = 0.0

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
        return s

    def __call__(self, arc_preds, rel_preds, arc_golds, rel_golds, mask):
        lens = mask.sum(1)
        arc_mask = arc_preds.eq(arc_golds) & mask
        rel_mask = rel_preds.eq(rel_golds) & arc_mask
        arc_mask_seq, rel_mask_seq = arc_mask[mask], rel_mask[mask]

        self.n += len(mask)
        self.n_ucm += arc_mask.sum(1).eq(lens).sum().item()
        self.n_lcm += rel_mask.sum(1).eq(lens).sum().item()

        self.total += len(arc_mask_seq)
        self.correct_arcs += arc_mask_seq.sum().item()
        self.correct_rels += rel_mask_seq.sum().item()
        return self

    @property
    def score(self):
        return self.las

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def uas(self):
        return self.correct_arcs / (self.total + self.eps)

    @property
    def las(self):
        return self.correct_rels / (self.total + self.eps)


class SpanMetric(Metric):

    def __init__(self, eps=1e-12):
        super().__init__()

        self.n = 0.0
        self.n_ucm = 0.0
        self.n_lcm = 0.0
        self.utp = 0.0
        self.ltp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        for pred, gold in zip(preds, golds):
            upred = Counter([(i, j) for i, j, label in pred])
            ugold = Counter([(i, j) for i, j, label in gold])
            utp = list((upred & ugold).elements())
            lpred = Counter(pred)
            lgold = Counter(gold)
            ltp = list((lpred & lgold).elements())
            self.n += 1
            self.n_ucm += len(utp) == len(pred) == len(gold)
            self.n_lcm += len(ltp) == len(pred) == len(gold)
            self.utp += len(utp)
            self.ltp += len(ltp)
            self.pred += len(pred)
            self.gold += len(gold)
        return self

    def __repr__(self):
        s = f"UCM: {self.ucm:6.2%} LCM: {self.lcm:6.2%} "
        s += f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} "
        s += f"LP: {self.lp:6.2%} LR: {self.lr:6.2%} LF: {self.lf:6.2%}"

        return s

    @property
    def score(self):
        return self.lf

    @property
    def ucm(self):
        return self.n_ucm / (self.n + self.eps)

    @property
    def lcm(self):
        return self.n_lcm / (self.n + self.eps)

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def lp(self):
        return self.ltp / (self.pred + self.eps)

    @property
    def lr(self):
        return self.ltp / (self.gold + self.eps)

    @property
    def lf(self):
        return 2 * self.ltp / (self.pred + self.gold + self.eps)


class ChartMetric(Metric):

    def __init__(self, eps=1e-12):
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)


class SeqTagMetric(Metric):
    def __init__(self, label_vocab, eps=1e-12):
        super().__init__()

        self.label_tp_lst = [0.0]*len(label_vocab)
        self.label_pd_lst = [0.0]*len(label_vocab)
        self.label_gd_lst = [0.0]*len(label_vocab)

        self.label_vocab = label_vocab
        self.tp = 0.0
        self.sum_num = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        span_mask = golds.ge(0)
        self.sum_num += span_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item()
        for i in range(len(self.label_vocab)):
            gold_mask = golds.eq(i) 
            pred_mask = preds.eq(i)
            self.label_gd_lst[i] += gold_mask.sum().item()
            self.label_pd_lst[i] += pred_mask.sum().item()
            self.label_tp_lst[i] += (pred_mask & gold_mask & span_mask).sum().item()

        return self

    def __repr__(self):
        s = ''
        for i in range(len(self.label_vocab)):
            label = self.label_vocab[i]
            p = self.label_tp_lst[i] / (self.label_pd_lst[i] + self.eps)
            r = self.label_tp_lst[i] / (self.label_gd_lst[i] + self.eps)
            f = 2*self.label_tp_lst[i] / (self.label_pd_lst[i] + self.eps + self.label_gd_lst[i])
            s += f"{label}: P:{p:6.2%} R:{r:6.2%} F:{f:6.2%} "
        return f"Accuracy: {self.accuracy:6.2%}"

    @property
    def score(self):
        return self.accuracy

    @property
    def accuracy(self):
        return self.tp / (self.sum_num + self.eps)

class AishellNerMetric(Metric):
    """
    string based (中国, LOC)
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.tp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps
        self.per_regex = '\[.+?\]'
        self.loc_regex = '\(.+?\)'
        self.org_regex = '\<.+?\>'

    def collect_nes(self, preds, pred_txt):
        assert len(preds) == len(pred_txt)
        pred_res = []
        for i in range(len(preds)):
            bio_lst = preds[i]
            snt = pred_txt[i]
            this_res = set()
            st = 0
            while st < len(bio_lst):
                curr_label = bio_lst[st]
                # find the first B- label
                if curr_label == 'O' or curr_label.startswith('I'):
                    st += 1
                    continue
                # take the label of B-label as the real label
                label = curr_label[2:]
                point = st + 1
                while point < len(bio_lst) and bio_lst[point].startswith('I'):
                    point += 1
                this_res.add((label, snt[st: point]))
                # this_res.add((label, st, point-1))
                st = point
            pred_res.append(this_res)
        return pred_res

    def collect_nes_from_e2e(self, preds):
        res = []
        for pred in preds:
            per_set = set([('PER', item.group()[1:-1]) for item in re.finditer(self.per_regex, pred)])
            loc_set = set([('LOC', item.group()[1:-1]) for item in re.finditer(self.loc_regex, pred)])
            org_set = set([('ORG', item.group()[1:-1]) for item in re.finditer(self.org_regex, pred)])
            res.append(per_set | loc_set | org_set)
        
        return res

    # def e2ecall(self, preds, golds, gold_txt):
    #     """
    #     preds: ["<中原地产>首席分析师[张大伟]说", ...]
    #     golds: [[O, O, B-LOC, O, O, B-ORG, I-ORG],
    #             [O, O, B-LOC, O, O, B-ORG, I-ORG]]
    #     """
    #     pred_res = self.collect_nes_from_e2e(preds)
    #     gold_res = self.collect_nes(golds, gold_txt)
    #     assert len(pred_res) == len(gold_res)
    #     for pred_set, gold_set in zip(pred_res, gold_res):
    #         self.pred += len(pred_set)
    #         self.gold += len(gold_set)
    #         self.tp += len(pred_set & gold_set)
    #     return self

    def e2ecall(self, preds, golds):
        """
        preds: ["<中原地产>首席分析师[张大伟]说", ...]
        golds: ["<中原地产>首席分析师[张大伟]说", ...]
        """
        pred_res = self.collect_nes_from_e2e(preds)
        gold_res = self.collect_nes_from_e2e(golds)
        assert len(pred_res) == len(gold_res)
        for pred_set, gold_set in zip(pred_res, gold_res):
            self.pred += len(pred_set)
            self.gold += len(gold_set)
            self.tp += len(pred_set & gold_set)
        return self


    def __call__(self, preds, golds, pred_txt, gold_txt):
        """
        preds: [[O, O, B-LOC, O, O, B-ORG, I-ORG],
                [O, O, B-LOC, O, O, B-ORG, I-ORG]]
        """
        pred_res = self.collect_nes(preds, pred_txt)
        gold_res = self.collect_nes(golds, gold_txt)
        assert len(pred_res) == len(gold_res)
        for pred_set, gold_set in zip(pred_res, gold_res):
            self.pred += len(pred_set)
            self.gold += len(gold_set)
            self.tp += len(pred_set & gold_set)

        return self

    @property
    def score(self):
        return self.f

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

    def __repr__(self):
        return f"P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"
