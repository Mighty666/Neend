# neendai experiment log

keeping track of everything i tried. lots of failures before stuff worked lol

---

## september 2023

### sep 15 - project start

started looking into sleep apnea after my dad got diagnosed. found out 80% of cases are undiagnosed. thats crazy. started reading papers about using audio for detection.

key papers found:
- nakano 2014 - tracheal sounds, 94% auc
- montazeri 2012 - snore acoustics correlate with ahi
- ng 2008 - formant frequencies distinguish osa

hypothesis: can i build something that works with just a phone microphone?

### sep 20 - dataset exploration

downloaded sleep-edf from physionet. 197 overnight recordings with psg labels. spent like 3 hours figuring out how to read the edf format.

first spectrogram looks cool! can clearly see snoring events as horizontal bands.

### sep 25 - first model attempt

tried simple cnn on mel spectrograms:
- 3 conv layers
- global avg pooling
- 2 fc layers

results: 68% accuracy, 0.72 auroc

not great but at least its learning something. loss is decreasing so thats good.

problems:
- class imbalance (only 10% apnea events)
- spectrograms look noisy

---

## october 2023

### oct 3 - adding features

added mfcc features based on speech recognition papers. concatenated with spectrogram before fc layers.

results: 75% accuracy, 0.78 auroc

better! mfccs definitely help. also added weighted loss for class imbalance which helped a lot.

### oct 10 - more data

found shhs dataset - 5000+ recordings! but its huge (like 500gb) so just downloaded a subset of 800 for now.

also found physionet apnea-ecg challenge data.

total: sleep-edf (197) + shhs (800) + physionet (450) = 1447 recordings

### oct 15 - denoising experiments

bedroom audio is NOISY. tried:
1. spectral subtraction - helped a little
2. wiener filter - about the same
3. bandpass filter 50-2000hz - removed some noise but also some signal

went with spectral subtraction for now. snr improved from ~15db to ~25db on average.

### oct 22 - feature engineering deep dive

implemented all the features from the papers:
- spectral centroid, bandwidth, rolloff, flatness
- zero crossing rate, rms energy
- teager energy (this was hard!)
- jitter, shimmer

ran correlation analysis with ahi:
- breathing pause duration: r=0.71 (wow!)
- snore spectral slope: r=0.54
- pitch variance: r=0.38

the pause duration feature alone gives like 0.85 auroc. makes sense - longer pauses = more severe apnea.

### oct 28 - resnet attempt

tried resnet18 pretrained on imagenet, fine-tuned on spectrograms.

results: 82% accuracy, 0.85 auroc

best so far but still not great. imagenet pretraining probably doesnt help much for audio.

---

## november 2023

### nov 5 - transformer idea

read about wav2vec 2.0 and got excited. the idea of pretraining on unlabeled audio then fine-tuning makes a lot of sense for medical applications where labels are expensive.

started implementing but its complicated. lots of moving parts:
- feature encoder (convolutions)
- quantizer (gumbel softmax)
- context network (transformer)
- contrastive loss

### nov 12 - pretraining struggles

spent the whole week trying to get pretraining to work. kept getting nan losses. turns out i needed:
1. gradient clipping
2. learning rate warmup
3. proper weight initialization

finally seeing the loss decrease! its slow though (~24 hours per 100k steps).

### nov 18 - first pretrained model working!

trained for 200k steps on all my unlabeled audio (~400 hours after augmentation).

fine-tuned on labeled data:
results: 86% accuracy, 0.89 auroc

NICE! pretraining actually helps. the model must be learning something useful about audio structure.

### nov 25 - scaling up

hypothesis: bigger model = better results

tried different sizes:
- 6 layers, 512 hidden: 0.89 auroc
- 12 layers, 768 hidden: 0.91 auroc
- 24 layers, 1024 hidden: 0.93 auroc

more layers definitely helps but training time is crazy. 24 layers takes like a week on my gpu.

also tried longer pretraining (400k steps) which helped a bit.

---

## december 2023

### dec 3 - masked modeling

switched from contrastive to masked modeling after reading the mae paper. simpler to implement and supposedly works just as well.

key insight: you can mask 75% of patches and still learn good representations. this makes training 4x faster!

results after retraining: 0.94 auroc

this is getting close to the clinical home sleep test performance (~0.95).

### dec 10 - more datasets

added a3 dataset (250 recordings) and cosmos (147 recordings).

total: 2847 recordings from 5 different sources

tested cross-dataset generalization:
- train on sleep-edf, test on shhs: 0.90 auroc (drops 4%)
- train on all, test on held-out: 0.93 auroc (drops 1%)

model generalizes reasonably well across datasets. this is important for real-world use.

### dec 15 - statistical rigor

my mentor reminded me i need proper statistical analysis for isef. implemented:
- bootstrap confidence intervals (1000 resamples)
- delong test for auroc comparison
- mcnemar test for accuracy comparison

95% ci for auroc: (0.935, 0.949)

this shows the result is robust and not just luck.

### dec 20 - hyperparameter search

ran optuna for 3 days straight (~1200 trials).

best hyperparameters:
- lr: 3e-4
- batch size: 32
- dropout: 0.15
- layers: 24
- hidden: 1024

final auroc: 0.942 ± 0.007

### dec 28 - audio signature validation

tested all 14 signatures from literature:

| signature | paper effect | my effect | validated? |
|-----------|--------------|-----------|------------|
| snore slope | r=0.58 | r=0.54 | yes |
| pause duration | auc=0.94 | auc=0.91 | yes |
| formant f1 | auc=0.82 | auc=0.79 | yes |
| pitch | r=-0.62 | r=-0.58 | yes |

all signatures replicated! this gives me confidence the model is learning real patterns.

---

## january 2024

### jan 5 - subgroup analysis

tested on different demographics:
- age <40: 0.951 auroc
- age 40-60: 0.938 auroc
- age >60: 0.927 auroc
- male: 0.940 auroc
- female: 0.945 auroc
- bmi >30: 0.931 auroc

performance is pretty consistent across groups. slight drop for older patients but still good.

### jan 10 - calibration

realized i should check if probabilities are calibrated. plot showed slight overconfidence for high probabilities.

applied temperature scaling (T=1.3) which improved ece from 0.052 to 0.032.

### jan 15 - streamlit dashboard

built interactive dashboard for visualizing results:
- upload audio recording
- see spectrogram, events, risk scores
- play back flagged segments

this will be useful for the isef demo.

### jan 20 - final results

**main result: 94.2% auroc for apnea detection**

comparison to baselines:
- cnn-resnet: 0.895 (p<0.001 vs ours)
- random forest: 0.823 (p<0.001)
- literature features only: 0.789 (p<0.001)

comparison to clinical methods:
- psg (gold standard): ~0.99
- home sleep test: ~0.95
- **ours: 0.942**

we're approaching home sleep test performance using just audio!

### jan 25 - documentation

writing up everything for isef:
- abstract (250 words)
- research paper (20 pages)
- poster
- lab notebook (this doc)

---

## key learnings

1. **pretraining matters**: self-supervised pretraining gave biggest improvement (+8% auroc)

2. **scale helps**: bigger model = better results, up to a point

3. **features matter**: literature-derived features are useful even with deep learning

4. **statistics are important**: need confidence intervals and significance tests

5. **generalization is hard**: cross-dataset performance drops but still acceptable

## failed experiments

1. lstm on raw waveform - too slow and didnt converge
2. spectral gating denoising - removed too much signal
3. imagenet pretrained vit - worse than random init (domain too different)
4. small batch size (8) - unstable training
5. no warmup - nan losses

## future work

1. prospective validation with home recordings
2. smartphone deployment
3. integration with wearables
4. longitudinal tracking

---

*total development time: ~4 months*
*total compute: ~2000 gpu hours (estimated)*
*total coffee: too much*
