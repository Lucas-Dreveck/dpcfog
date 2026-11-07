# dpcfog: pre-filtering malicious recommendations with Density Peaks Clustering

Code and results for the undergraduate thesis *Pré-Filtragem de Recomendações
Maliciosas em Fog-IoT Utilizando Density Peaks Clustering* (Faculdade
Donaduzzi, 2026), written in Portuguese.

The work asks whether Density Peaks Clustering (DPC), run on a Fog node, can
discard malicious recommendations (bad-mouthing, ballot-stuffing, collusion)
before they reach the Trust Score. Its accuracy and cost are compared with six
baselines, from a static consensus heuristic to two supervised neural models.
Neither hypothesis held: DPC did not lead on any real network, and its
interval contains chance on all three. This repository holds everything needed
to reproduce that result.

## Three evaluation layers

| Layer | Data | Ground truth | Supports claims about |
|---|---|---|---|
| 1 | Bitcoin OTC, Bitcoin Alpha, Epinions | published per-user fraud labels (Kumar et al., 2018) | filters on real trust networks |
| 2 | Bitcoin OTC with injected profiles (Mobasher et al., 2007) | the injection itself | the same, with the caveat of the injection arrangement |
| 3 | Fog-IoT simulation | per recommendation | mechanism only, never operation |

The real networks record trade and opinion between people, not recommendations
exchanged by IoT devices. Moving any layer-1 result to Fog-IoT is a hypothesis.

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest tests -q                    # 58 tests; data-dependent ones skip without data

python -m realdata.download                  # networks and labels into ./data
python -m realdata.real_experiment otc alpha # layer 1
python -m pytest tests -q                    # now with the real-data tests
```

`DPCFOG_DATA` points to another data folder. Python 3.12.10 and the exact
versions in `requirements.txt` were used for every result; PyTorch runs on CPU.

## Guided tour

`walkthrough.ipynb` rebuilds the numbers of the paper's Results section from
the CSVs in `results/`, section by section, and checks each one against the
value printed in the paper; a value that moves beyond rounding fails its cell.
It needs no downloaded data and no PyTorch, only pandas and matplotlib, and
runs in a few seconds:

```bash
python -m pip install pandas==3.0.5 matplotlib==3.11.0 ipykernel==7.4.0 nbconvert==7.17.2
jupyter nbconvert --to notebook --execute walkthrough.ipynb   # or open it in any Jupyter frontend
```

Its last section lists the numbers that need the raw networks, with the
command for each.

## Where each result of the paper comes from

| Result | Command | Output |
|---|---|---|
| Table 1, networks | `python -m realdata.download` (prints the checked counts) | stdout |
| Table 3, layer-1 AUC with 95% CI by rater | `python paper_tables.py --redo` | `outputs/table_layer1.tex` |
| Table 4, AUC against mu (simulation) | `python run_grid.py`, then `python paper_tables.py` | `results/full_grid.csv`, `outputs/table_mu.tex` |
| Layer-1 F1, discard rate, Trust Score error | `python -m realdata.real_experiment otc alpha epinions` | `results/real*.csv` |
| Layer-2 F1 and AUC under both labels | `python -m realdata.attack_experiment otc` | `results/attack_otc.csv` |
| Out-of-distribution AUC of the trained models | `python out_of_distribution.py` | `results/out_of_distribution.csv` |
| Cost against batch size, Raspberry Pi 3 budget | `python scalability.py` | stdout |
| Memory by two instruments; cost with 1, 4, 16 threads | `python -m analysis.memory_instruments`, `python -m analysis.cost_by_threads` | `outputs/*.json` |
| Consensus capture | `python -m analysis.consensus_capture otc` | stdout |
| Feature-weighting ceiling | `python -m analysis.feature_weighting otc` | stdout |
| Anchors versus clustering in the Han et al. adaptation | `python -m analysis.han_ablation --sweep` | stdout |
| Paired differences with intervals, new versus seen raters, independent H2 reference, Deep Trust epochs | `python -m analysis.layer1_predictions`, then `python -m analysis.paired_comparisons` | `outputs/paired_comparisons_*.json` |
| Train/evaluation split per layer | `python -m analysis.temporal_split` | stdout |
| Per-rater versus per-recommendation intervals | `python -m analysis.ci_by_rater` | `outputs/ci_by_rater.json` |
| DPC validation against Rodriguez & Laio (2014) | `python -m validation.graded`, `python -m validation.rodriguez2014` | stdout |
| REV2 against Table 3 of Kumar et al. (2018) | `python -m validation.rev2_kumar2018` | stdout |
| Shuffled-label and oracle controls | `python -m validation.controls` | stdout |
| Committed CSV still matches the code | `python -m validation.stale_csv` | stdout |

Run every command from the repository root.

## Parameters

Everything below is a default in the code; nothing was tuned on evaluation
data. Cuts are fixed and identical across networks, seeds and scenarios.

### DPC (`dpcfog/filters/dpc.py`)

| Parameter | Value |
|---|---|
| Features | signed deviation from the target median; rater mean; rater variance; rater agreement (share within 0.10 of the consensus); consistency (std of the rater's mean deviation across rounds). Standardised per batch |
| Density | Gaussian kernel |
| d_c | 2% percentile of the positive pairwise distances (Rodriguez & Laio, 2014) |
| Centres | largest jump in log(gamma), gamma = rho x delta normalised, at most 8, density peak always a centre |
| Decision | discard everything outside the cluster holding the density peak, plus the halo of that cluster |
| Variants | `dpc_entropy` (d_c by minimum potential entropy), `dpc_cardinality` (largest cluster is legitimate), `dpc_recommender` (cluster raters, not recommendations), `dpc_outlier` (discard if delta > d_c, no partition), `dpc_bimodal` (features weighted by Sarle's bimodality coefficient) |
| Han et al. adaptation | 1-D values plus two anchors at 0 and 1, K = 3, d_c at 2%, per target; `dpc_han_no_id` drops the neighbour id coordinate |

### Baselines

| Filter | Parameters |
|---|---|
| Consensus | discard if absolute deviation > 0.25 |
| Rater history (Boukerch et al., 2007) | reputation = 0.8 previous + 0.2 agreement (tolerance 0.15), no optimistic prior, discard below 0.5 |
| REV2 (Kumar et al., 2018) | 15 iterations, tolerance 1e-4, gamma_1 = gamma_2 = 1, gamma_3 = alpha = beta = 0, per batch, discard reliability < 0.5 |
| Depth-3 tree | scikit-learn `DecisionTreeClassifier(max_depth=3)` |
| Neural network | scikit-learn `MLPClassifier`, hidden layers 64-32-16, ReLU, Adam, 200 iterations, no early stopping, cut at probability 0.5 |
| Deep Trust (Ullah et al., 2024) | PCA keeping 95% variance, one GRU layer of 64 units, 8-step rater sequence, Adam 1e-3, 40 epochs, batch 256, class-weighted loss, cut at probability 0.5 |

Supervised models train only on labelled ratings of the training rounds; there
is no separate validation set. Seed 0 in layer 1, the run seed elsewhere.

### Layers 1 and 2 (`realdata/`)

| Parameter | Value |
|---|---|
| Rounds | monthly, one Fog node; Epinions daily with 8 Fog nodes |
| History window | 5 rounds; batches with fewer than 10 ratings are dropped |
| Split | first 3 rounds discarded; of the rest, the first 40% train, the remaining 60% evaluate |
| Trust Score reference | mean rating from users labelled benign (layer 1); mean real rating before injection (layer 2) |
| Attacks (layer 2) | random (bad-mouthing, ballot-stuffing), average (stealthy), bandwagon and segment (collusion); one attack target; profile size drawn from the real degree distribution; mu 0.10 to 0.40; seeds 0 to 2 |

### Layer 3 (`dpcfog/config.py`, `dpcfog/scenarios.py`)

| Parameter | Value |
|---|---|
| Network | 100 nodes, 2 Fog nodes (nodes split at random in equal halves), 50 rounds |
| Recommendations | each node recommends K targets drawn among the others per round (collusion members aim at shared victims) |
| Scenarios | `calibrated` (benign T* in [0.57, 0.68], malicious in [0.10, 0.49], sigma 0.08, K = 3, estimated on Bitcoin OTC), `basic` (T* benign [0.70, 1.00], malicious [0, 0.30], sigma 0.05, K = 10), `adverse` (basic with per-node sigma 0.02 to 0.20, stealthy deviation 0.15 to 0.30, collusion on 5% of benign nodes) |
| Attacks | bad-mouthing (reports 0.3), ballot-stuffing (0.9), collusion (half of the attackers coordinated, a third of the noise, half the slots on 15% of benign nodes), On-Off (attacks 40% of a 10-round cycle), mixed |
| Split | warm-up rounds 0 to 4, training 5 to 22, evaluation 23 to 49 |
| Grid | mu 0.10 to 0.40, seeds 0 to 9: 600 runs per filter |

## Layout

```
dpcfog/        simulation, features, filters, Trust Score, metrics, experiment loop
realdata/      real networks: download, ingestion, batches, attack injection, layers 1-2
validation/    DPC and REV2 against their papers, controls, stale-CSV check
analysis/      measurements behind specific statements of the Results section
tests/         58 tests
walkthrough.ipynb  the paper's numbers from results/, each checked against the paper
results/       committed CSVs: the evidence behind the paper's numbers
paper_tables.py, run_grid.py, scalability.py, out_of_distribution.py
```

## Decisions that change how results read

- The filter runs per (Fog node, round), never on the whole network. DPC is
  O(n^2) in time and memory; on Epinions with a single node the mean batch
  would hold 9,575 ratings and need 700 MB, so the Fog partition becomes a
  condition for running at all.
- Features come from a history window. A Fog node in operation has no global
  view, and a number measured with information that would not exist in
  production does not transfer.
- In real data, unknown is not benign. Only 23.5% of the Bitcoin OTC ratings
  come from labelled users. The `evaluable` mask decides what the metric
  counts; the filter processes everything, as it would in operation.
- Time and memory are measured in separate passes. `tracemalloc` slows the run
  down, and it does not see PyTorch allocations, which is why process memory
  is measured too.
- F1 never appears alone. Discard rates range from 0.02% (REV2 on Epinions) to
  78% (`dpc_recommender` on Epinions), so F1 also compares the calibration of
  each cut; AUC and average precision measure the signal without the
  threshold.
- Layer-1 intervals resample raters. The label belongs to the user, and the
  ratings of one user are not independent.

## Data

The networks and labels are not redistributed: `realdata.download` fetches them
from SNAP (https://snap.stanford.edu/data/) and from the REV2 page
(https://cs.stanford.edu/~srijan/rev2/), and ingestion fails unless the counts
match Table 2 of Kumar et al. (2018). The committed CSVs hold aggregate metrics
only, with no user identifier.

## References

- Boukerch, A., Xu, L., and El-Khatib, K. (2007). Trust-based security for wireless ad hoc and sensor networks. *Computer Communications*, 30(11-12).
- Han, Y., Wang, H., Li, Y., and Zhang, L. (2024). Trust-aware and improved density peaks clustering algorithm for fast and secure models in wireless sensor networks. *Pervasive and Mobile Computing*, 105.
- Kumar, S., Hooi, B., Makhija, D., Kumar, M., Faloutsos, C., and Subrahmanian, V. S. (2018). REV2: Fraudulent user prediction in rating platforms. *WSDM*.
- Mobasher, B., Burke, R., Bhaumik, R., and Williams, C. (2007). Toward trustworthy recommender systems: An analysis of attack models and algorithm robustness. *ACM Transactions on Internet Technology*, 7(4).
- Rodriguez, A. and Laio, A. (2014). Clustering by fast search and find of density peaks. *Science*, 344(6191).
- Ullah, F., Salam, A., Amin, F., Khan, I. A., Ahmed, J., Zaib, S. A., et al. (2024). Deep Trust: A novel framework for dynamic trust and reputation management in the Internet of Things (IoT)-based networks. *IEEE Access*, 12.

## License

MIT; see `LICENSE`. To cite, see `CITATION.cff`.
