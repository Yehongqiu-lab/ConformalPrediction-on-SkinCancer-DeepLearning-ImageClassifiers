''' CP.py runs on the env: sklearn'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold

class CP(object):    
    def __init__(self, md_name, alpha=0.1, num_class=7, R=1000,
                 sc=0, lam_reg=0, k_reg=0, disallow_zero_set=True, rand=True,# scoring method is APS in default (0) and RAPS with sc=1
                 **kwargs):
   
        self.md_name = md_name
        self.a = alpha
        self.num_class = num_class
        self.R = R
        
        if sc==1:
            self.updt_hypas(k_reg, lam_reg)
            self.disallow_zero_set = disallow_zero_set
            self.rand = rand
            
        if "smx" in kwargs and "labels" in kwargs and "n_cali" in kwargs:
             self.smx = kwargs["smx"]
             self.labels = kwargs["labels"] 
             self.n = kwargs["n_cali"]
             self.split_cali()
        elif "cal_smx" in kwargs and "val_smx" in kwargs and "cal_labels" in kwargs and "val_labels" in kwargs:
            self.cal_smx, self.val_smx, self.cal_labels, self.val_labels = \
                kwargs["cal_smx"], kwargs["val_smx"], kwargs["cal_labels"], kwargs["val_labels"]
            self.n = self.cal_smx.shape[0]
        else:
            raise ValueError("cannot process the input type!")
                
    ## Basic Operations
    def split_cali(self):
        idx = np.array([1] * self.n + [0] * (self.smx.shape[0] - self.n)) > 0
        np.random.shuffle(idx)
        self.cal_smx, self.val_smx = self.smx[idx, :], self.smx[~idx, :]
        self.cal_labels, self.val_labels = self.labels[idx], self.labels[~idx]

    def get_sample_coverage(self, pt=True, **kwargs):
        if "prediction_sets" in kwargs and "val_labels" in kwargs:
            prediction_sets = kwargs["prediction_sets"]
            val_labels = kwargs["val_labels"]
        else:
            prediction_sets = self.prediction_sets
            val_labels = self.val_labels
        sample_coverage = prediction_sets[
                np.arange(prediction_sets.shape[0]), val_labels
                                                   ].mean()
        if pt == False: return sample_coverage
        print(f"The empirical coverage is: {sample_coverage}")
        return sample_coverage
    
    def hist_SPS(self, plot=True):
        set_size = np.sum(self.prediction_sets, axis=1)
        if plot == False: return np.round(set_size.mean(), 2)
        sns.histplot(set_size, discrete=True)
        plt.title(
            f"Histogram of Prediction Set Size of {self.md_name} \n(with average {np.round(set_size.mean(), 2)}))")
        plt.xlabel("Prediction Set Size")
        plt.gca().set_xticks(np.arange(0, self.num_class+1, 1))
        plt.ylabel("Frequency")
        plt.show()

    def get_fsc(self, array=True):
        idx = {}
        fsc_array = np.zeros(self.num_class)
        n_val = self.val_labels.shape[0]
        for i in range(self.num_class):
            idx[f"{i}"] = np.arange(n_val)[self.val_labels == i]
        fsc = np.inf
        for i in range(self.num_class):
           tmp = self.prediction_sets[idx[f"{i}"], i].mean()
           fsc_array[i] = tmp
           if tmp < fsc: fsc = tmp
        if array == False: return fsc
        print(f"The feature-stratified coverage array is:\n {fsc_array}")  
        return fsc
    
    def get_ssc(self, array=True):
        idx = {"1":[], "2":[], "3":[]}
        ssc_array = np.zeros(3)
        n_val = self.val_labels.shape[0]
        idx["1"] = np.arange(n_val)[np.sum(self.prediction_sets, axis=1) == 1]
        idx["2"] = np.arange(n_val)[np.sum(self.prediction_sets, axis=1) == 2]
        idx["3"] = np.arange(n_val)[np.sum(self.prediction_sets, axis=1) > 2]
        ssc = np.inf
        for i in range(3):
            tmp = self.prediction_sets[idx[f"{i+1}"], self.val_labels[idx[f"{i+1}"]]].mean()
            ssc_array[i] = tmp
            if tmp < ssc: ssc = tmp
        if array == False: return ssc
        print(f"The size-stratified coverage array is:\n {ssc_array}")
        return ssc
    
    def cc(self, scores): # correctness check
        # calculate the coverage R times and store in list
        coverages = np.zeros((self.R,))
        
        for r in range(self.R):
            np.random.shuffle(scores) # shuffle
            calib_scores, val_scores = (scores[:self.n],scores[self.n:]) # split
            qhat = np.quantile(calib_scores, np.ceil((self.n+1)*(1-self.a))/self.n, interpolation='higher') # calibrate
            coverages[r] = (val_scores <= qhat).astype(float).mean()
        average_coverage = coverages.mean() # should be close to 1-alpha

        sns.histplot(coverages)
        plt.title(f"Histogram of Empirical Coverages over {self.R} trails \n(with average {np.round(average_coverage, 2)})")
        plt.xlabel("set size")
        plt.gca().set_xticks(np.arange(0.8, 1.09, .1))
        plt.ylabel("Frequency")
        plt.show()
    
    ## Scoring Method 1: Adaptive Prediction Set (APS)
    def get_APS_score(self, heu_score, lbs):
        pi = heu_score.argsort(1)[:, ::-1]
        srt = np.take_along_axis(heu_score, pi, axis=1).cumsum(axis=1)
        scores = np.take_along_axis(srt, pi.argsort(axis=1), axis=1)[
            range(heu_score.shape[0]), lbs]
        return scores
    
    def get_APS(self):
        cal_scores = self.get_APS_score(self.cal_smx, self.cal_labels)
        qhat = np.quantile(
            cal_scores, np.ceil((self.n + 1) * (1 - self.a)) / self.n,
            interpolation="higher")
        val_pi = self.val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(self.val_smx, val_pi, axis=1).cumsum(axis=1)
        self.prediction_sets = np.take_along_axis(val_srt <= qhat, 
                                                  val_pi.argsort(axis=1), axis=1)
        return self.prediction_sets
    
    ## Scoring Method 2: Regularized Adaptive Prediction Sets (RAPS)
    def updt_hypas(self, k_reg, lam_reg):
        self.k_reg = k_reg
        self.lam_reg = lam_reg
        self.reg_vec = np.array(self.k_reg*[0,] + 
                                    (self.num_class-self.k_reg)*[self.lam_reg,])[None,:]

    def get_RAPS_score(self, **kwargs):
        if "k_reg" in kwargs and "lam_reg" in kwargs:
            self.updt_hypas(kwargs["k_reg"], kwargs["lam_reg"])  
        if "heu_score" in kwargs and "lbs" in kwargs:
            heu_score, lbs = kwargs["heu_score"], kwargs["lbs"]
        else:
            heu_score, lbs = self.cal_smx, self.cal_labels
            
        n = heu_score.shape[0]
        pi = heu_score.argsort(1)[:, ::-1]
        srt = np.take_along_axis(heu_score, pi, axis=1)
        srt_reg = srt + self.reg_vec
        L = np.where(pi == lbs[:, None])[1] # the ground-truth column indices in the ordered sfx matrix (or the huescore mtx)
        scores = srt_reg.cumsum(axis=1)[np.arange(n),L] - np.random.rand(n)*srt_reg[np.arange(n),L]
        return scores

    def get_RAPS(self, **kwargs):
        flag = 0
        if "k_reg" in kwargs and "lam_reg" in kwargs:
            self.updt_hypas(kwargs["k_reg"], kwargs["lam_reg"])
        if "heu_score" in kwargs and "lbs" in kwargs and "val_smx" in kwargs:
            heu_score, lbs, val_smx = kwargs["heu_score"], kwargs["lbs"], kwargs["val_smx"]
        else:
            heu_score, lbs, val_smx = self.cal_smx, self.cal_labels, self.val_smx
            flag = 1
            
        n = heu_score.shape[0]
        cal_scores = self.get_RAPS_score(heu_score=heu_score, lbs=lbs)
        qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - self.a)) / n, 
                           interpolation="higher")
        n_val = val_smx.shape[0]
        val_pi = val_smx.argsort(1)[:,::-1]
        val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
        val_srt_reg = val_srt + self.reg_vec
        indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat \
            if self.rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
        if self.disallow_zero_set: indicators[:,0] = True
        prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)
        if flag == 1: 
            self.prediction_sets = prediction_sets
        return prediction_sets

    def cv_hypas(self, n_splits=10):
        lambda_grid = [1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 0.001, 0.005, 0.01, 0.05]  # Values to test for lam_reg
        k_grid = [0, 1, 2, 3]  # Values to test for k_reg
        kf = KFold(n_splits=n_splits, shuffle=True)
        results = []

        for lam_reg in lambda_grid:
            for k_reg in k_grid:
                fold_coverages = []
                fold_set_sizes = []
        
                for train_idx, cvval_idx in kf.split(self.cal_smx):
                    train_smx, cvval_smx = self.cal_smx[train_idx], self.cal_smx[cvval_idx]
                    train_labels, cvval_labels = self.cal_labels[train_idx], self.cal_labels[cvval_idx]
            
                    # Compute prediction sets and empirical coverage
                    prediction_sets = self.get_RAPS(heu_score=train_smx, lbs=train_labels, val_smx=cvval_smx, k_reg=k_reg, lam_reg=lam_reg)
                    empirical_coverage = self.get_sample_coverage(pt=False, 
                                                                  prediction_sets=prediction_sets, 
                                                                  val_labels=cvval_labels)
                    fold_coverages.append(empirical_coverage)
                    # Compute average set size
                    prediction_sets_size = prediction_sets.sum(axis=1)
                    average_set_size = np.mean(prediction_sets_size)
                    fold_set_sizes.append(average_set_size)
        
                # Store results
                avg_coverage = np.mean(fold_coverages)
                avg_set_size = np.mean(fold_set_sizes)
                results.append({
                    "lam_reg": lam_reg,
                    "k_reg": k_reg,
                    "coverage": avg_coverage,
                    "set_size": avg_set_size
                })
        
            
        # Select optimal parameters
        valid_results = [res for res in results if res["coverage"]-1+self.a >= -0.25]
        optimal_params = min(valid_results, key=lambda x: x["set_size"]) if valid_results else None

        if optimal_params:
            print(f"Optimal parameters: lam_reg={optimal_params['lam_reg']}, k_reg={optimal_params['k_reg']}")
            print(f"Coverage: {optimal_params['coverage']}, Average set size: {optimal_params['set_size']}")
            self.updt_hypas(optimal_params["k_reg"], optimal_params["lam_reg"])
        else:
            print("No valid parameters found that meet the desired coverage.")