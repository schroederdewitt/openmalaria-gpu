# import pyro
from collections import defaultdict
from torch.distributions import normal, poisson, binomial, log_normal, categorical, uniform
import random
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from torch import nn

# physical constants
alpha_m = 0.9
alpha_m_star = 2.53
A_max = 1.63
D_x = 0.0
E_star = 0.032
gamma_p = 2.04
S_imm = 0.14
S_infty = 0.049
sigma_i = 10.2**0.5
sigma_0 = 0.66**0.5
X_nu_star = 4.8
X_p_star = 1514.4
X_h_star = 97.3
X_y_star = 3.5
Y_h_star = 9999999 #float("inf")

# simulation constants
delta = 5
EIR_scenario = "Namawala"
is_Garki = True  # not sure!
nu = 4.8 if is_Garki else 0.18
parasite_detection_limit = 2 if is_Garki else 40


# Download EIRs - daily EIRs
import re
import requests

#EIR_urls = {"Namawala":
#                "https://raw.githubusercontent.com/SwissTPH/openmalaria/6e5207cce791737c53d97d51be584f76d0059447/test/scenario5.xml"}
#EIR_dict = {}
#for name, url in EIR_urls.items():
#    text = requests.get(url)
#    res = re.findall(r"<EIRDaily.*>(.*)<.*", text.content.decode("utf-8"))
#    EIR_dict[name] = [float(x) for x in res]

# from https://www.worldometers.info/demographics/nigeria-demographics/#age-structure
age_distributions = {"Nigeria": th.FloatTensor([[0, 4, 0.1646],
                                                [5, 9, 0.1451],
                                                [10, 14, 0.1251],
                                                [15, 19, 0.1063],
                                                [20, 24, 0.0877],
                                                [25, 29, 0.00733],
                                                [30, 34, 0.00633],
                                                [35, 39, 0.00549],
                                                [40, 44, 0.00459],
                                                [45, 49, 0.00368],
                                                [50, 54, 0.00288],
                                                [55, 59, 0.00231],
                                                [60, 64, 0.00177],
                                                [65, 69, 0.00126],
                                                [70, 74, 0.00083],
                                                [75, 79, 0.00044],
                                                [80, 84, 0.00016],
                                                [85, 89, 0.00004]]),
                     "DUMMY25": th.FloatTensor([[24, 25, 1.0]]),
                     }

# Select and adapt EIR frequency
def change_EIR_frequency(EIRdaily, delta):
    EIR = []
    EIR.append(sum(EIRdaily[-delta:]))
    for t in range(0, len(EIRdaily)-1, delta):
        EIR.append(sum(EIRdaily[t:t+delta]))
    return EIR
#EIR = change_EIR_frequency(EIR_dict[EIR_scenario], delta)
# EIR = [0.0206484514, 0.058492964, 0.20566511399999998, 0.30665830299999997, 0.5819757700000001,
#        0.9119642099999999, 0.9812269200000001, 1.08515392, 1.5639562, 1.91511741, 2.26906343,
#        1.4642980899999998, 1.44800599, 0.4953665689999999, 0.188470482, 0.12150698499999998, 0.18865241400000002,
#        0.185076822, 0.139661401, 0.175435914, 0.208139087, 0.234284155, 0.30769259, 0.298616083, 0.351398984,
#        0.262847315, 0.23533705099999996, 0.143308623, 0.329922543, 0.30471578899999996, 0.334684268, 0.267825345,
#        0.10680065749999999, 0.11492165600000001, 0.129193927, 0.10540250799999999, 0.11679969899999999, 0.097755679,
#        0.10671757, 0.0618278874, 0.0647416485, 0.038036469499999996, 0.0377843451, 0.0364936385, 0.0380708517,
#        0.047425256, 0.0326014605, 0.0489408695, 0.0666497766, 0.0296905653, 0.06196254500000001, 0.0623980334,
#        0.047184591000000005, 0.036532315999999995, 0.052737068, 0.0421134431, 0.0394260218, 0.0141218558,
#        0.014938396999999999, 0.00644923895, 0.0095492285, 0.0249317087, 0.0320313153, 0.0132738001,
#        0.022837353400000003, 0.09629449899999999, 0.106729029, 0.23377416750000002, 0.34699540500000003,
#        0.15817682300000002, 0.179243571, 0.131176548, 0.042414276, 0.0206484514]


########################################################################################################################
########################################################################################################################
########################################################################################################################


class PopulationWarmup(nn.Module):

    def __init__(self, n, sim_params, age_distribution, EIR, use_cache=True, device=None):

        self.simulation_params = sim_params
        self.use_cache = use_cache
        self.device = device
        self.n = n
        self.age_dist = age_distribution.to(self.device)
        self.EIR = th.from_numpy(EIR).to(self.device)
        pass


    def forward(self, record=False):
        self.record = record
        if record:
            self.init_recording()

        # set up simulation parameters
        delta = self.simulation_params["delta"]

        # set up physical parameters
        alpha_m = self.simulation_params["alpha_m"]
        alpha_m_star = self.simulation_params["alpha_m_star"]

        # draw initial population max ages (and sort them)
        max_age_bands = categorical.Categorical(self.age_dist[: ,2]).sample((n,))
        max_ages = (uniform.Uniform(self.age_dist[: ,0][max_age_bands],
                                   self.age_dist[: ,1][max_age_bands]).sample()*365).int().sort(descending=True)[0]

        # schedule = self._bin_pack(max_ages)

        # draw host variations
        log_d = th.log(log_normal.LogNormal(th.tensor(0.0).to(self.device),
                                     th.tensor(sigma_i).to(self.device)).sample((n,)))

        max_age = max_ages[0].item()

        # create tensor storage for intermediate values
        X_h = th.zeros((n,), device=self.device).float()
        X_p = th.zeros((n,), device=self.device).float()
        X_y_infidx = None
        Y = th.zeros((n,), device=self.device).float()

        # create list storage for infection events
        tau_infidx = None # later: Tensor, 0: tau_infidx_0, 1: tau_infidx_max, 2: pop id

        # create offset index tensor
        offset_idxs = max_age - max_ages

        # pre-cache effective EIR
        t_coords = th.arange(0, max_age, delta, device=self.device).long()
        eff_EIR = self.EIR[((t_coords%365)//delta)] *self._body_surface(t_coords.float()) / A_max

        # pre-cache D_m
        D_m = 1.0 - alpha_m *th.exp(-((t_coords.float()/365.0)/alpha_m_star ) *th.log(th.tensor(2.0)))

        bidx = 0 # index indicating whether a member of the population has already been born

        if self.record:
            items = {"max_ages": max_ages.clone(),
                     "log_d": log_d.clone(),
                     }
            self.recs[0].update(items)

        # we should employ a bin-packing scheduler here! https://en.wikipedia.org/wiki/Bin_packing_problem
        # this would ensure we optimally exploit parallelism at all times!
        for t in range(0, max_age, delta):

            # update idx determining whether people have already been born at time t
            while bidx < n and max_age - max_ages[bidx] <= t:
                bidx += 1

            # relative time idxs
            rel_idxs = (t - offset_idxs[:bidx]).long()

            E = eff_EIR.repeat(len(rel_idxs))[rel_idxs // delta]

            # calculate force of infection
            h_star = self._force_of_infection(t, E, X_p[:bidx], Y[:bidx])

            # generate infections
            tau_infidx_new = self._new_infections(t, h_star)
            if tau_infidx_new is not None:

                # update infections
                if tau_infidx is None:
                    tau_infidx = tau_infidx_new
                    X_y_infidx = th.zeros((len(tau_infidx),), device=self.device)
                else:
                    tau_infidx = th.cat([tau_infidx, tau_infidx_new])
                    X_y_infidx = th.cat([X_y_infidx,
                                         th.zeros((len(tau_infidx_new),), device=self.device)])

            ## discard infections that are already over, and attach new ones

            # update immunity
            ret = self._update_immunity(log_d[:bidx],
                                        t,
                                        rel_idxs,
                                        h_star,
                                        tau_infidx,
                                        X_y_infidx,
                                        X_h[:bidx],
                                        D_m[rel_idxs//delta])
            Y[:bidx], X_y_infidx, X_h[:bidx] = ret

            # Discard expired infections
            if tau_infidx is not None:
                mask = tau_infidx[:, 1]>t
                tau_infidx = tau_infidx[mask].clone()
                X_y_infidx = X_y_infidx[mask].clone()

            X_p[:bidx] += E
            if self.record:
                items = {"X_p": X_p.clone(),
                         "Y": Y.clone(),
                         "X_y": X_y_infidx if X_y_infidx is None else X_y_infidx.clone() ,
                         "X_h": X_h.clone(),
                         "tau": tau_infidx if tau_infidx is None else tau_infidx.clone(),
                         "h_star": h_star if h_star is None else h_star.clone(),
                         "E": E
                         }
                self.recs[t].update(items)

        return Y, X_y_infidx, X_h

    def _bin_pack(self, lens):
        from sortedcontainers import SortedList
        slens = sorted([l.item() for l in lens], reverse=True)
        bins = SortedList(key=lambda x: slens[0] - sum(x))
        for l in slens:
            if not bins:
                bins.add([l])
                continue
            idx = bins.bisect_right([slens[0] - l])
            if idx >= len(bins):
                bins.add([l])
            else:
                n = bins[idx]
                bins.discard(n)
                bins.add(n + [l])
            y = [slens[0] - sum(x) for x in bins]
            f = 0
        return bins

    def init_recording(self):
        self.recs = defaultdict(lambda: {})

    def backward(self):

        pass

    def _update_immunity(self, log_d, t, t_idxs, h_star, tau_infidx, X_y_infidx, X_h, D_m):
        sigma_0 = self.simulation_params["sigma_0"]
        X_y_star = self.simulation_params["X_y_star"]
        X_h_star = self.simulation_params["X_h_star"]
        X_nu_star = self.simulation_params["X_nu_star"]

        if tau_infidx is None or not len(tau_infidx): # no current infections anywhere!
            Y = th.zeros(log_d.shape, device=self.device)
            return Y, X_y_infidx, X_h

        # Update
        sigma_y = sigma_0 *((1 + X_h /X_nu_star )**(-0.5))
        D_h = (1 + X_h/X_h_star )**(-1)
        D_y_infidx = (1 + X_y_infidx/X_y_star )**(-1)

        ##################################################################################
        # Now we have to convert all relevant quantities from agent idxs to infection idxs
        in_idxs = tau_infidx[:, 2].long()
        log_d_infidx = log_d[in_idxs]
        D_h_infidx = D_h[in_idxs]
        t_infidx = t_idxs[in_idxs]
        D_m_infidx = D_m[in_idxs]
        sigma_y_infidx = sigma_y[in_idxs]

        # Calculate concurrent infections
        if S_infty < float("inf"):
            M, M_infidx = self._groupby_aggregate_sum(in_idxs,
                                                      th.ones(in_idxs.shape, device=self.device),
                                                      dim_size=len(log_d))
        else:
            M, M_infidx = None, None

        # Update parasite densities per infections
        y_infidx = th.exp(self._get_ln_parasite_density(log_d_infidx,
                                                         #t_infidx,
                                                         t,
                                                         tau_infidx,
                                                         D_y_infidx,
                                                         D_h_infidx,
                                                         D_m_infidx,
                                                         M_infidx,
                                                         sigma_y_infidx))

        # Update total parasite densities (see https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335)
        Y, Y_infidx = self._groupby_aggregate_sum(in_idxs,
                                                  y_infidx,
                                                  dim_size=len(log_d))

        if th.any(Y != Y):
            a = 5
            pass

        # Update
        X_h = X_h + h_star

        if self.record:
            items = {"sigma_y":sigma_y,
                     "D_h":D_h,
                     "M": M,
                     "D_m": D_m
                     }
            self.recs[t].update(items)

        # Update immunity due to pre-erythrocytic exposure # TODO: UNSURE ABOUT THIS HERE!
        X_y_infidx = X_y_infidx + Y_infidx - y_infidx

        if not th.all(th.isfinite(X_y_infidx)) or not th.all(th.isfinite(y_infidx)):
            a = 5
            pass

        return Y, X_y_infidx, X_h

    def _force_of_infection(self, t, E, X_p, Y):
        """
        Batch-ready
        """
        S_1 = S_infty + ( 1-S_infty ) /( 1 + E /E_star)
        S_2 = S_imm + ( 1-S_imm ) /( 1 +(X_p /X_p_star )**gamma_p)

        S_p = S_1 *S_2
        _lambda = S_p *E
        # S_h = ( 1 + Y /Y_h_star)**(-1)
        h = poisson.Poisson(_lambda).sample()
        h_star = h.clone().long()
        h_mask = (h!=0.0)
        # if th.any(S_h < 1.0) and th.sum(h_mask) > 0:
        #    try:
        #        h_star[h_mask] = binomial.Binomial(h[h_mask].float(), S_h[h_mask]).sample().long()
        #    except Exception as e:
        #        a = 5
        #        pass

        if self.record:
            items = {"S_1":S_1,
                     "S_2":S_2}
            self.recs[t].update(items)
        return h_star


    def _new_infections(self, t, h):
        """
        Batch-ready
        """
        tot_h = h.sum().int().item()
        if tot_h == 0:
            return None
        tau_infidx = th.zeros((tot_h, 3), device=self.device)
        tau_infidx[: ,0] = t
        tau_infidx[:, 1] = t + th.exp(normal.Normal(5.13, 0.8).sample((tot_h,)))
        tau_infidx[: ,2] = th.repeat_interleave(th.arange(len(h), device=self.device), h)
        return tau_infidx

    pass

    def _ln_y_G(self, t_infidx, tau_infidx):
        """
        Batch-ready
        """
        delta = self.simulation_params["delta"]
        tau_infidx_0 = tau_infidx[:, 0]
        tau_infidx_max = tau_infidx[:, 1]
        a = 0.018 *(tau_infidx_max-tau_infidx_0)
        a[a>4.4] = 4.4
        c = a/ (1 + (tau_infidx_max - tau_infidx_0) / 35.0)
        b = th.log(a / c) / (tau_infidx_max - tau_infidx_0)
        ln_y = a * th.exp(-b * (t_infidx - tau_infidx_0)) - nu
        ln_y[ln_y<=0.0] = 10**(-10)
        # replace exponential with some tricks, i.e. log(n+m) = log(m) + log(1+m/n)
        #temp = (-b * (t_infidx - tau_infidx_0))
        #term = b.clone().zero_()
        #term[temp > np.log(nu)] = th.log(1-nu/th.exp(temp[temp > np.log(nu)]))
        #ln_y = temp.clone().zero_() - 10**10
        #ln_y[temp > np.log(nu)] = th.log(a[temp > np.log(nu)]) + temp[temp > np.log(nu)] + term[temp > np.log(nu)]
        #y[y<=0] = 10E-8 # entirely undetectable
        #y[t_infidx < delta] = 1.0
        #if th.any(ln_y>20):
        #    a = 5
        #    pass
        #return ln_y #th.log(y)
        return ln_y

    def _get_ln_parasite_density(self,
                                 log_d_infidx,
                                 t_infidx,
                                 tau_infidx,
                                 D_y_infidx,
                                 D_h_infidx,
                                 D_m_infidx,
                                 M_infidx,
                                 sigma_y_infidx):
        """
        Batch-ready
        """
        E_ln_y_0 = log_d_infidx + self._ln_y_G(t_infidx, tau_infidx)
        E_ln_y = D_y_infidx * D_h_infidx * D_m_infidx * E_ln_y_0 + th.log(D_x / M_infidx + 1 - D_x)
        ln_y_tau_infidx = normal.Normal(E_ln_y, sigma_y_infidx).sample()
        if not th.all(th.isfinite(ln_y_tau_infidx)):
            a = 5
            pass
        return ln_y_tau_infidx

    def _groupby_aggregate_sum(self, labels, samples, dim_size=None):
        """
        :param labels: 1D array
        :param samples: 2D array (batch x value)
        :return:

        NOTE: This should be replaced by scatter_add from
        https://github.com/rusty1s/pytorch_scatter/blob/master/README.md
        as current method uses too much memory +  is not performant enough!
        """
        # minl = labels.min().long().item() if min_label is None else min_label
        #maxl = labels.max().long().item() if max_label is None else max_label
        #A = th.zeros((maxl - minl + 1), len(samples))
        #A[(labels-minl).long(), th.arange(len(samples))] = 1
        #X = th.mm(A, samples).squeeze(1)
        #B = th.sum(A, dim=1).long()
        from torch_scatter import scatter_add
        X = scatter_add(samples, labels.long(), dim_size=dim_size)
        X_infidx = X[labels.long()]
        return X, X_infidx

    def _body_surface(self, age):
        """
        Batch-ready
        """
        age = age.clone()
        age[age > 20 * 365] = 20 * 365
        age /= 365.0
        b = (2.0097152973788390) * 10 ** (-1)
        c = (1.7984405384216584) * 10 ** (-1)
        d = (-2.1215879908575791) * 10 ** (-2)
        e = (1.3554013908681018) * 10 ** (-3)
        f = (-2.8476973535248611) * 10 ** (-5)
        A = b + c * age + d * (age ** 2) + e * (age ** 3) + f * (age ** 4)
        return A

########################################################################################################################
########################################################################################################################
########################################################################################################################


if __name__ == "__main__":


    # from torch_scatter import scatter_add
    #
    # src = th.Tensor([[2, 0, 1, 4, 3]])
    # #index = th.tensor([[4, 5, 4, 2, 3]])
    # index = th.tensor([[4, 4, 4, 2, 3]])
    # # out = src.new_zeros((2, 6))
    #
    # out = scatter_add(src, index, dim_size=6)
    #
    # print(out)
    # quit()

    #def _safe_log_sum(log_a, log_b):
    #    """
    #    :param log_a:
    #    :param log_b:
    #    :return: log(a+b)
    #    """
    #    res = log_a
    #    res[log_a> ]th.log(1 + th.exp(log_b)/th.exp(log_a))


    device = "cuda:0" # "cuda:0" #"cuda:0" #"#"cpu" #"cuda:0" # "cuda:0"
    n = 1000
    delta = 5
    EIR_scenario = "Namawala" #"Namawala"
    #EIR = change_EIR_frequency(EIR_dict[EIR_scenario], delta)
    EIR = [0.0206484514, 0.058492964, 0.20566511399999998, 0.30665830299999997, 0.5819757700000001,
           0.9119642099999999, 0.9812269200000001, 1.08515392, 1.5639562, 1.91511741, 2.26906343,
           1.4642980899999998, 1.44800599, 0.4953665689999999, 0.188470482, 0.12150698499999998, 0.18865241400000002,
           0.185076822, 0.139661401, 0.175435914, 0.208139087, 0.234284155, 0.30769259, 0.298616083, 0.351398984,
           0.262847315, 0.23533705099999996, 0.143308623, 0.329922543, 0.30471578899999996, 0.334684268, 0.267825345,
           0.10680065749999999, 0.11492165600000001, 0.129193927, 0.10540250799999999, 0.11679969899999999, 0.097755679,
           0.10671757, 0.0618278874, 0.0647416485, 0.038036469499999996, 0.0377843451, 0.0364936385, 0.0380708517,
           0.047425256, 0.0326014605, 0.0489408695, 0.0666497766, 0.0296905653, 0.06196254500000001, 0.0623980334,
           0.047184591000000005, 0.036532315999999995, 0.052737068, 0.0421134431, 0.0394260218, 0.0141218558,
           0.014938396999999999, 0.00644923895, 0.0095492285, 0.0249317087, 0.0320313153, 0.0132738001,
           0.022837353400000003, 0.09629449899999999, 0.106729029, 0.23377416750000002, 0.34699540500000003,
           0.15817682300000002, 0.179243571, 0.131176548, 0.042414276, 0.0206484514]
    EIR = np.array(EIR) # if EIR is extremely high, then can get NANS!!
    is_Garki = False  # not sure!
    nu = 4.8 if is_Garki else 0.18
    parasite_detection_limit = 2 if is_Garki else 40

    sim_params = {"delta": delta,
                  "alpha_m": alpha_m,
                  "alpha_m_star": alpha_m_star,
                  "A_max": A_max,
                  "D_x": D_x,
                  "E_star": E_star,
                  "gamma_p": gamma_p,
                  "S_imm": S_imm,
                  "S_infty": S_infty,
                  "sigma_i": sigma_i,
                  "sigma_0": sigma_0,
                  "X_nu_star": X_nu_star,
                  "X_p_star": 1514.4,
                  "X_h_star": 97.3,
                  "X_y_star": 3.5,
                  "Y_h_star": float("inf"),
                  "parasite_detection_limit": parasite_detection_limit,
                  "nu": nu}

    # set up initial population distribution for Nigeria
    # age_dist = age_distributions["Nigeria"].to(device)
    age_dist = age_distributions["DUMMY25"].to(device)

    import time
    pw = PopulationWarmup(n, sim_params, age_dist, EIR, device = device)

    t = time.time()
    ret = pw.forward(record=True)
    delta_t = time.time() - t
    print("TOTAL TIME: {}s ".format(delta_t))
    print("TIME PER YEAR: {}s ".format(delta_t/25.0)) # only works if DUMMY25!
    print("TIME PER THOUSAND PER YEAR: {}".format(delta_t/(25.0 * n)))

    print(pw.recs[0].keys())
    print("finished!")
    #quit() # DEBUG


    # gather the data
    n_years = 20
    t_idx = np.array(range(0, n_years*365, 5))/365.0
    #i = 0

    fig, ax = plt.subplots(nrows=13, sharex=True, figsize=(12, 20))

    for i in range(min(n, 10)):
        Y_lst = [pw.recs[t]["Y"][i] if i < len(pw.recs[t]["Y"]) else 0.0 for t in sorted(pw.recs.keys())]
        Xp_lst = [pw.recs[t]["X_p"][i] if i < len(pw.recs[t]["X_p"]) else 0.0 for t in sorted(pw.recs.keys())]
        log_d = pw.recs[0]["log_d"][i]
        max_ages = pw.recs[0]["max_ages"][i]/365.0
        h_star = [pw.recs[t]["h_star"][i] if i < len(pw.recs[t]["h_star"]) else 0.0 for t in sorted(pw.recs.keys())]
        n_infs = [pw.recs[t]["M"][i] if ("M" in pw.recs[t]) and (i < len(pw.recs[t]["M"])) else 0.0 for t in sorted(pw.recs.keys())]
        sigma_y = [pw.recs[t]["sigma_y"][i] if ("sigma_y" in pw.recs[t]) and i < len(pw.recs[t]["sigma_y"]) else 0.0 for t in sorted(pw.recs.keys())]
        S_1 = [pw.recs[t]["S_1"][i] if i < len(pw.recs[t]["S_1"]) else 0.0 for t in sorted(pw.recs.keys())]
        S_2 = [pw.recs[t]["S_2"][i] if i < len(pw.recs[t]["S_2"]) else 0.0 for t in sorted(pw.recs.keys())]
        D_m = [pw.recs[t]["D_m"][i] if ("D_m" in pw.recs[t]) and i < len(pw.recs[t]["D_m"]) else 0.0 for t in sorted(pw.recs.keys())]
        D_h = [pw.recs[t]["D_h"][i] if ("D_h" in pw.recs[t]) and i < len(pw.recs[t]["D_h"]) else 0.0 for t in sorted(pw.recs.keys())]
        E = [pw.recs[t]["E"][i] if ("E" in pw.recs[t]) and i < len(pw.recs[t]["E"]) else 0.0 for t in
               sorted(pw.recs.keys())]

        ax[0].plot(t_idx, Y_lst[:len(t_idx)], label="total parasite density (t)")
        ax[1].plot(np.array(range(0, n_years*365, 5))/365.0, n_infs[:len(t_idx)], label="concurrent infections (t)")
        ax[2].plot(np.array(range(0, n_years*365, 5))/365.0, h_star[:len(t_idx)], label="h_star(t)")
        ax[3].plot(np.array(range(0, n_years*365, 5))/365.0,
                   np.array(np.cumsum(np.array(Y_lst[:len(t_idx)]))), label="Cumulative Parasite Densities")
        ax[4].plot(np.array(range(0, n_years*365, 5))/365.0, Xp_lst[:len(t_idx)], label="Cumulative Exposure")
        ax[5].plot(np.array(range(0, n_years*365, 5))/365.0, sigma_y[:len(t_idx)], label="$\sigma_y$")
        ax[6].plot(np.array(range(0, n_years*365, 5))/365.0, S_1[:len(t_idx)], label="$S_1$")
        ax[7].plot(np.array(range(0, n_years*365, 5))/365.0, S_2[:len(t_idx)], label="$S_2$")
        ax[8].plot(np.array(range(0, n_years*365, 5))/365.0, D_m[:len(t_idx)], label="$D_m$")
        ax[9].plot(np.array(range(0, n_years*365, 5))/365.0, D_h[:len(t_idx)], label="$D_h$")
        ax[10].plot(np.array(range(0, n_years*365, 5))/365.0, np.array(E[:len(t_idx)]), label="EIR(t)")
        ax[11].plot(np.array(range(0, n_years*365, 5))/365.0, np.array([log_d]*(n_years*365//5)), label="$d_i$")
        ax[12].plot(np.array(range(0, n_years*365, 5))/365.0, np.array([max_ages]*(n_years*365//5)), label="$d_i$")

    ax[0].set_title("Total Parasite Density $Y(t)$ [parasites/$\mu L$]")
    ax[1].set_title("Concurrent Infections $M(t)$")
    ax[2].set_title("New Infections $h^*(t)$")
    ax[3].set_title("Cumulative Parasite Densities [parasites/$\mu L\cdot$days]")
    ax[4].set_title("Cumulative Exposure $X_p$(t)")
    ax[5].set_title("Density variations between hosts including effects of cumulative exposure $\sigma_y$")
    ax[6].set_title("Success probability of innoculations $S_1$")
    ax[7].set_title("Immunity effect due to Acquired pre-erythrocytic immunity $S_2$")
    ax[8].set_title("Immunity effect on parasite densities due to maternal immunity $D_m$")
    ax[9].set_title("Immunity effect on parasite densities due to cumulative number of prior infections $D_h$")
    ax[10].set_title("Effective EIR(t)")
    ax[11].set_title("Force of infection variations between hosts $d_i$")

    ax[3].set_xlabel("Age (years)")
    ax[0].set_yscale("log")
    ax[3].set_yscale("log")
    ax[11].set_yscale("log")
    from matplotlib.ticker import MaxNLocator
    ax[1].yaxis.set_major_locator(MaxNLocator(integer=True))
    ax[2].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.show()
    fig.savefig("warmup_graph.pdf", bbox_inches='tight')