# pyright: reportMissingImports=true, reportUntypedBaseClass=false, reportGeneralTypeIssues=False

import numpy as np
import matplotlib.pyplot as plt

class black_box_benchmarks(object):
    
    def __init__(self, shadow_train_performance, shadow_test_performance, target_train_performance, target_test_performance,
                 num_classes):
        '''
        each input contains both model predictions (shape: num_data*num_classes) and ground-truth labels. 
        Modification: correct labels passed separately incase of mislabelling.
        '''
        self.num_classes = num_classes
        
        self.s_tr_outputs, self.s_tr_labels = shadow_train_performance
        self.s_te_outputs, self.s_te_labels = shadow_test_performance
        self.t_tr_outputs, self.t_tr_labels = target_train_performance
        self.t_te_outputs, self.t_te_labels = target_test_performance
        # print(f's_tr_labels: {self.s_tr_labels}')
        # print(f's_te_labels: {self.s_te_labels}')
        # print(f't_tr_labels: {self.t_tr_labels}')
        # print(f't_te_labels: {self.t_te_labels}')
        
        # self.s_tr_corr = (np.argmax(self.s_tr_outputs, axis=1)==self.s_tr_labels).astype(int)
        # self.s_te_corr = (np.argmax(self.s_te_outputs, axis=1)==self.s_te_labels).astype(int)
        # self.t_tr_corr = (np.argmax(self.t_tr_outputs, axis=1)==self.t_tr_labels).astype(int)
        # self.t_te_corr = (np.argmax(self.t_te_outputs, axis=1)==self.t_te_labels).astype(int)
        
        self.s_tr_conf = np.array([self.s_tr_outputs[i, self.s_tr_labels[i]] for i in range(len(self.s_tr_labels))])
        print(f's_tr_conf: {self.s_tr_conf}')
        self.s_te_conf = np.array([self.s_te_outputs[i, self.s_te_labels[i]] for i in range(len(self.s_te_labels))])
        self.t_tr_conf = np.array([self.t_tr_outputs[i, self.t_tr_labels[i]] for i in range(len(self.t_tr_labels))])
        self.t_te_conf = np.array([self.t_te_outputs[i, self.t_te_labels[i]] for i in range(len(self.t_te_labels))])
        
        # self.s_tr_entr = self._entr_comp(self.s_tr_outputs)
        # print(f's_tr_entr: {self.s_tr_entr}')
        # self.s_te_entr = self._entr_comp(self.s_te_outputs)
        # self.t_tr_entr = self._entr_comp(self.t_tr_outputs)
        # self.t_te_entr = self._entr_comp(self.t_te_outputs)
        
        # self.s_tr_m_entr = self._m_entr_comp(self.s_tr_outputs, self.s_tr_labels)
        # print(f's_tr_m_entr: {self.s_tr_m_entr}')
        # self.s_te_m_entr = self._m_entr_comp(self.s_te_outputs, self.s_te_labels)
        # self.t_tr_m_entr = self._m_entr_comp(self.t_tr_outputs, self.t_tr_labels)
        # self.t_te_m_entr = self._m_entr_comp(self.t_te_outputs, self.t_te_labels)
        
    
    def _log_value(self, probs, small_value=1e-30):
        return -np.log(np.maximum(probs, small_value))
    
    def _entr_comp(self, probs):
        return np.sum(np.multiply(probs, self._log_value(probs)),axis=1)
    
    def _m_entr_comp(self, probs, true_labels):
        log_probs = self._log_value(probs)
        reverse_probs = 1-probs
        log_reverse_probs = self._log_value(reverse_probs)
        modified_probs = np.copy(probs)
        modified_probs[range(true_labels.size), true_labels] = reverse_probs[range(true_labels.size), true_labels]
        modified_log_probs = np.copy(log_reverse_probs)
        modified_log_probs[range(true_labels.size), true_labels] = log_probs[range(true_labels.size), true_labels]
        return np.sum(np.multiply(modified_probs, modified_log_probs),axis=1)
    
    def _thre_setting(self, tr_values, te_values):
        value_list = np.concatenate((tr_values, te_values))
        print(f'mean tr_value: {np.mean(tr_values)} \t mean te_value: {np.mean(te_values)}')
        tr_sz, te_sz = len(tr_values), len(te_values)
        thre, max_acc = 0, 0
        for value in value_list:
            tr_ratio = np.sum(tr_values>=value)/(tr_sz+0.0)
            te_ratio = np.sum(te_values<value)/(te_sz+0.0)
            acc = 0.5*(tr_ratio + te_ratio)
            if acc > max_acc:
                thre, max_acc = value, acc
                print(f'tr_sz: {tr_sz}\tte_sz: {te_sz}\t-\tthre: {thre}\tnew_best_acc: {max_acc}\t')
        return thre, max_acc
    
    def _mem_inf_via_corr(self):
        # perform membership inference attack based on whether the input is correctly classified or not
        s_tr_acc = np.sum(self.s_tr_corr)/(len(self.s_tr_corr)+0.0)
        s_te_acc = np.sum(self.s_te_corr)/(len(self.s_te_corr)+0.0)
        shadow_acc = 0.5*(s_tr_acc + 1 - s_te_acc)
        t_tr_acc = np.sum(self.t_tr_corr)/(len(self.t_tr_corr)+0.0)
        t_te_acc = np.sum(self.t_te_corr)/(len(self.t_te_corr)+0.0)
        mem_inf_acc = 0.5*(t_tr_acc + 1 - t_te_acc)
        print('For membership inference attack via correctness, the attack acc is {acc1:.3f}, with train acc {acc2:.3f} and test acc {acc3:.3f}'.format(acc1=mem_inf_acc, acc2=t_tr_acc, acc3=t_te_acc) )
        return shadow_acc, mem_inf_acc
    
    def _mem_inf_thre(self, v_name, s_tr_values, s_te_values, t_tr_values, t_te_values):
        # perform membership inference attack by thresholding feature values: the feature can be prediction confidence,
        # (negative) prediction entropy, and (negative) modified entropy
        shadow_acc, t_tr_mem, t_te_non_mem = 0, 0, 0
        for num in range(self.num_classes):
            print(f'class number: {num}')
            s_tr_values_curr, s_te_values_curr = s_tr_values[self.s_tr_labels==num], s_te_values[self.s_te_labels==num]
            samples_curr = len(s_tr_values_curr) + len(s_te_values_curr)
            if samples_curr==0: continue 
            thre, shadow_acc_curr = self._thre_setting(s_tr_values_curr, s_te_values_curr)
            shadow_acc += shadow_acc_curr*samples_curr
            t_tr_mem += np.sum(t_tr_values[self.t_tr_labels==num]>=thre)
            t_te_non_mem += np.sum(t_te_values[self.t_te_labels==num]<thre)
        shadow_acc /= (len(s_tr_values) + len(s_te_values))
        mem_inf_acc = 0.5*(t_tr_mem/(len(self.t_tr_labels)+0.0) + t_te_non_mem/(len(self.t_te_labels)+0.0))
        print('For membership inference attack via {n}, the attack acc is {acc:.3f}'.format(n=v_name,acc=mem_inf_acc))
        return shadow_acc, mem_inf_acc
    
    def _mem_inf_benchmarks(self, all_methods=True, benchmark_methods=[]):
        benchmarks = {"corr_sha":0, "corr_tar":0, "conf_sha":0, "conf_tar":0,
                        "entr_sha":0, "entr_tar":0, "mentr_sha":0, "mentr_tar":0}
        if (all_methods) or ('correctness' in benchmark_methods):
            benchmarks["corr_sha"], benchmarks["corr_tar"] = self._mem_inf_via_corr()
        if (all_methods) or ('confidence' in benchmark_methods):
            benchmarks["conf_sha"], benchmarks["conf_tar"] =\
                 self._mem_inf_thre('confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf, self.t_te_conf)
        if (all_methods) or ('entropy' in benchmark_methods):
            benchmarks["entr_sha"], benchmarks["entr_tar"] =\
                 self._mem_inf_thre('entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr, -self.t_te_entr)
        if (all_methods) or ('modified entropy' in benchmark_methods):
            benchmarks["mentr_sha"], benchmarks["mentr_tar"] =\
                 self._mem_inf_thre('modified entropy', -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr, -self.t_te_m_entr)

        return benchmarks


def distrs_compute(tr_values, te_values, tr_labels, te_labels, num_bins=5, log_bins=True, plot_name=None):
    
    ### function to compute and plot the normalized histogram for both training and test values class by class.
    ### we recommand using the log scale to plot the distribution to get better-behaved distributions.
    
    num_classes = len(set(tr_labels))
    sqr_num = np.ceil(np.sqrt(num_classes))
    tr_distrs, te_distrs, all_bins = [], [], []
    
    plt.figure(figsize = (15,15))
    plt.rc('font', family='serif', size=10)
    plt.rc('axes', linewidth=2)
    
    for i in range(num_classes):
        tr_list, te_list = tr_values[tr_labels==i], te_values[te_labels==i]
        if log_bins:
            # when using log scale, avoid very small number close to 0
            small_delta = 1e-10
            tr_list[tr_list<=small_delta] = small_delta
            te_list[te_list<=small_delta] = small_delta
        n1, n2 = np.sum(tr_labels==i), np.sum(te_labels==i)
        all_list = np.concatenate((tr_list, te_list))
        max_v, min_v = np.amax(all_list), np.amin(all_list)
        
        plt.subplot(sqr_num, sqr_num, i+1)
        if log_bins:
            bins = np.logspace(np.log10(min_v), np.log10(max_v),num_bins+1)
            weights = np.ones_like(tr_list)/float(len(tr_list))
            h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
            plt.gca().set_xscale("log")
            weights = np.ones_like(te_list)/float(len(te_list))
            h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
            plt.gca().set_xscale("log")
        else:
            bins = np.linspace(min_v, max_v,num_bins+1)
            weights = np.ones_like(tr_list)/float(len(tr_list))
            h1, _,_ = plt.hist(tr_list,bins=bins,facecolor='b',weights=weights,alpha = 0.5)
            weights = np.ones_like(te_list)/float(len(te_list))
            h2, _, _ = plt.hist(te_list,bins=bins,facecolor='r',weights=weights,alpha = 0.5)
        tr_distrs.append(h1)
        te_distrs.append(h2)
        all_bins.append(bins)
    if plot_name == None:
        plot_name='./tmp'
    plt.savefig(plot_name+'.png', bbox_inches='tight')
    tr_distrs, te_distrs, all_bins = np.array(tr_distrs), np.array(te_distrs), np.array(all_bins)
    return tr_distrs, te_distrs, all_bins


def risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels):
    
    ### Given training and test distributions (obtained from the shadow classifier), 
    ### compute the corresponding privacy risk score for training points (of the target classifier).
    
    def find_index(bins, value):
        # for given n bins (n+1 list) and one value, return which bin includes the value
        if value>=bins[-1]:
            return len(bins)-2 # when value is larger than any bins, we assign the last bin
        if value<=bins[0]:
            return 0  # when value is smaller than any bins, we assign the first bin
        return np.argwhere(bins<=value)[-1][0]
    
    def score_calculate(tr_distr, te_distr, ind): 
        if tr_distr[ind]+te_distr[ind] != 0:
            return tr_distr[ind]/(tr_distr[ind]+te_distr[ind])
        else: # when both distributions have 0 probabilities, we find the nearest bin with non-zero probability
            for t_n in range(1, len(tr_distr)):
                t_ind = ind-t_n
                if t_ind>=0:
                    if tr_distr[t_ind]+te_distr[t_ind] != 0:
                        return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                t_ind = ind+t_n
                if t_ind<len(tr_distr):
                    if tr_distr[t_ind]+te_distr[t_ind] != 0:
                        return tr_distr[t_ind]/(tr_distr[t_ind]+te_distr[t_ind])
                    
    risk_score = []   
    for i in range(len(data_values)):
        c_value, c_label = data_values[i], data_labels[i]
        c_tr_distr, c_te_distr, c_bins = tr_distrs[c_label], te_distrs[c_label], all_bins[c_label]
        c_index = find_index(c_bins, c_value)
        c_score = score_calculate(c_tr_distr, c_te_distr, c_index)
        risk_score.append(c_score)
    return np.array(risk_score)

def calculate_risk_score(tr_values, te_values, tr_labels, te_labels, data_values, data_labels, 
                         num_bins=5, log_bins=True):
    
    ########### tr_values, te_values, tr_labels, te_labels are from shadow classifier's training and test data
    ########### data_values, data_labels are from target classifier's training data
    ########### potential choice for the value -- entropy, or modified entropy, or prediction loss (i.e., -np.log(confidence))
    
    tr_distrs, te_distrs, all_bins = distrs_compute(tr_values, te_values, tr_labels, te_labels, 
                                                    num_bins=num_bins, log_bins=log_bins)
    risk_score = risk_score_compute(tr_distrs, te_distrs, all_bins, data_values, data_labels)
    return risk_score