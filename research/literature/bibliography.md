# bibliography

all the papers i read for this project. starred ones were really important.

## core datasets

⭐ kemp b, zwinderman ah, tuk b, kamphuisen hac, oberye jjl. analysis of a sleep-dependent neuronal feedback loop: the slow-wave microcontinuity of the eeg. ieee trans biomed eng. 2000;47(9):1185-1194. doi:10.1109/10.867928
- this is the sleep-edf database paper. used their data for training

⭐ quan sf, howard bv, iber c, et al. the sleep heart health study: design, rationale, and methods. sleep. 1997;20(12):1077-1085.
- shhs study design. huge dataset, took 2 weeks to get access

penzel t, moody gb, mark rg, goldberger al, peter jh. the apnea-ecg database. comput cardiol. 2000;27:255-258.
- physionet apnea-ecg database. small but classic benchmark

## self-supervised learning

⭐ baevski a, zhou h, mohamed a, auli m. wav2vec 2.0: a framework for self-supervised learning of speech representations. neurips 2020.
- the og ssl paper for audio. tried to implement this but masked modeling was easier

hsu wn, bolte b, tsai yh, lakhotia k, salakhutdinov r, mohamed a. hubert: self-supervised speech representation learning by masked prediction of hidden units. ieee/acm taslp. 2021;29:3451-3460.
- hubert paper. similar to wav2vec but uses k-means targets

⭐ niizumi d, takeuchi d, ohishi y, harada n, kashino k. byol for audio: self-supervised learning for general-purpose audio representation. ijcnn 2021.
- byol-a paper. really elegant approach, no negative samples needed

⭐ he k, chen x, xie s, li y, dollar p, girshick r. masked autoencoders are scalable vision learners. cvpr 2022.
- mae paper. this is what i ended up using for pretraining. 75% mask ratio works great

## statistical methods

⭐ efron b, tibshirani rj. an introduction to the bootstrap. chapman and hall/crc; 1993.
- the bootstrap bible. read chapters 1-5 and 12-14

delong er, delong dm, clarke-pearson dl. comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. biometrics. 1988;44(3):837-845.
- delong test for comparing aurocs. used this for all model comparisons

benjamini y, hochberg y. controlling the false discovery rate: a practical and powerful approach to multiple testing. j r stat soc b. 1995;57(1):289-300.
- fdr correction. important when you're running a lot of tests

## sleep apnea detection

ng ak, koh ts, baey e, lee th, abeyratne ur, kawahara k. could formant frequencies of snore signals be an alternative means for the diagnosis of obstructive sleep apnea? sleep med. 2008;9(8):894-898.
- formant analysis for apnea. implemented their f1-f4 extraction

abeyratne ur, wakwella as, hurber c. pitch jump probability measures for the analysis of snoring sounds in apnea. physiol meas. 2005;26(5):779-798.
- pitch analysis paper. the pitch jump stuff is interesting

⭐ fiz ja, jane r, sola-soler j, abad j, garcia ma, morera j. continuous analysis and monitoring of snores and their relationship to the apnea-hypopnea index. laryngoscope. 2010;120(4):854-862.
- really good paper on acoustic features for ahi correlation

## architecture stuff

su j, lu y, pan s, murtadha a, wen b, liu y. roformer: enhanced transformer with rotary position embedding. arxiv:2104.09864. 2021.
- rope paper. better than learned positional embeddings

shazeer n. glu variants improve transformer. arxiv:2002.05202. 2020.
- swiglu paper. small improvement over regular ffn

zhang b, sennrich r. root mean square layer normalization. neurips 2019.
- rmsnorm. faster than layernorm, basically the same performance

## other useful papers

guan h, liu m. domain adaptation for medical image analysis: a survey. ieee trans biomed eng. 2021.
- good overview of domain shift issues in medical ai

rudin c. stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. nat mach intell. 2019;1(5):206-215.
- made me think about interpretability. we should probably add more explainability features

## papers i should probably read but haven't yet

- attention is all you need (vaswani 2017) - read parts but should read whole thing
- bert (devlin 2019) - same deal
- vit (dosovitskiy 2020) - only skimmed it

---

total papers cited: 18
papers actually read: ~25
papers skimmed: ~100

last updated: jan 2024
