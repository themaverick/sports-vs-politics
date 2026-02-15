"""
Sports vs Politics Text Classifier
------------------------------------
Downloads the BBC News dataset, filters the 'sport' and 'politics'
categories, and compares four ML classifiers across three feature
representations.

Dataset : BBC News (Greene & Cunningham, ICML 2006)
           http://mlg.ucd.ie/datasets/bbc.html

Classifiers : Multinomial Naive Bayes, Logistic Regression,
              Linear SVM, Random Forest
Features    : Bag of Words, TF-IDF, Bigram TF-IDF

Usage:
    python b22ch045_prob4.py

Roll number: b22ch045
"""

import os
import sys
import zipfile
import urllib.request
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────
BBC_URL   = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbc_data")
CLASSES   = ["sport", "politics"]
LABEL_MAP = {"sport": 0, "politics": 1}
NAMES     = ["Sport", "Politics"]
SEED      = 42

# ─────────────────────────────────────────────
#  1. Data collection
# ─────────────────────────────────────────────

def download_bbc():
    """Download and extract the BBC News zip if not already present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    bbc_root = os.path.join(DATA_DIR, "bbc")

    if os.path.isdir(bbc_root) and os.listdir(bbc_root):
        print("  Dataset already present at", bbc_root)
        return bbc_root

    zip_path = os.path.join(DATA_DIR, "bbc-fulltext.zip")
    print("  Downloading from", BBC_URL)
    urllib.request.urlretrieve(BBC_URL, zip_path)
    print("  Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    return bbc_root


def load_articles(bbc_root):
    """Read all .txt articles from the sport/ and politics/ folders."""
    texts, labels = [], []
    for cat in CLASSES:
        folder = os.path.join(bbc_root, cat)
        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            if not os.path.isfile(path):
                continue
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                texts.append(text)
                labels.append(LABEL_MAP[cat])
    return texts, labels

# ─────────────────────────────────────────────
#  2. Dataset analysis
# ─────────────────────────────────────────────

def print_stats(texts, labels):
    """Print basic dataset statistics."""
    n = len(texts)
    n_sport = labels.count(0)
    n_pol   = labels.count(1)
    lengths = [len(t.split()) for t in texts]

    print("\n--- Dataset statistics ---")
    print("  Total documents : {}".format(n))
    print("  Sport           : {} ({:.1f}%)".format(n_sport, 100*n_sport/n))
    print("  Politics        : {} ({:.1f}%)".format(n_pol, 100*n_pol/n))
    print("  Avg doc length  : {:.0f} words".format(np.mean(lengths)))
    print("  Min / Max       : {} / {}".format(min(lengths), max(lengths)))


def plot_class_distribution(labels, out_dir):
    """Bar chart of class counts."""
    counts = [labels.count(0), labels.count(1)]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bars = ax.bar(NAMES, counts, color=["#2196F3", "#FF5722"], width=0.5)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width()/2, b.get_height()+3,
                str(c), ha="center", fontweight="bold")
    ax.set_ylabel("Documents")
    ax.set_title("Class Distribution")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "class_dist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved", path)


def plot_top_words(texts, labels, out_dir):
    """Horizontal bar chart of top-15 words per class."""
    sport_words  = Counter(w for i,t in enumerate(texts) if labels[i]==0
                           for w in t.lower().split() if len(w)>3)
    pol_words    = Counter(w for i,t in enumerate(texts) if labels[i]==1
                           for w in t.lower().split() if len(w)>3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (name, ctr, color) in enumerate([
        ("Sport", sport_words.most_common(15), "#2196F3"),
        ("Politics", pol_words.most_common(15), "#FF5722"),
    ]):
        words = [w for w,_ in ctr]
        freqs = [c for _,c in ctr]
        axes[idx].barh(words[::-1], freqs[::-1], color=color)
        axes[idx].set_title("Top 15 words - " + name)
        axes[idx].set_xlabel("Frequency")
        axes[idx].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "top_words.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved", path)

# ─────────────────────────────────────────────
#  3. Feature extraction
# ─────────────────────────────────────────────

def build_features(X_train, X_test):
    """
    Build three feature sets from raw text:
      - BoW         (CountVectorizer)
      - TF-IDF      (TfidfVectorizer)
      - Bigram TFIDF(TfidfVectorizer with ngram_range=(1,2))
    Returns dict  { name: (X_tr, X_te, vectorizer) }
    """
    feats = {}

    bow = CountVectorizer(lowercase=True, stop_words="english", max_features=10000)
    feats["BoW"] = (bow.fit_transform(X_train), bow.transform(X_test), bow)

    tfidf = TfidfVectorizer(lowercase=True, stop_words="english", max_features=10000)
    feats["TF-IDF"] = (tfidf.fit_transform(X_train), tfidf.transform(X_test), tfidf)

    bigram = TfidfVectorizer(lowercase=True, stop_words="english",
                             ngram_range=(1,2), max_features=15000)
    feats["Bigram"] = (bigram.fit_transform(X_train), bigram.transform(X_test), bigram)

    return feats

# ─────────────────────────────────────────────
#  4. Classifiers
# ─────────────────────────────────────────────

def get_classifiers():
    return {
        "Naive Bayes":  MultinomialNB(alpha=1.0),
        "Logistic Reg": LogisticRegression(max_iter=1000, C=1.0, random_state=SEED),
        "SVM (Linear)": SVC(kernel="linear", C=1.0, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED),
    }

# ─────────────────────────────────────────────
#  5. Training & evaluation
# ─────────────────────────────────────────────

def run_experiments(X_train, X_test, y_train, y_test):
    """Train every classifier on every feature set and collect results."""
    feats = build_features(X_train, X_test)
    clfs  = get_classifiers()
    results = []

    for feat_name, (Xtr, Xte, _) in feats.items():
        print("\n  --- {} ---".format(feat_name))
        for clf_name, clf in clfs.items():
            clf.fit(Xtr, y_train)
            y_pred = clf.predict(Xte)

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
            cm   = confusion_matrix(y_test, y_pred)

            results.append({
                "feature": feat_name, "clf": clf_name,
                "label": feat_name + " + " + clf_name,
                "acc": acc, "prec": prec, "rec": rec, "f1": f1, "cm": cm,
            })
            print("    {:<15s} Acc={:.4f}  P={:.4f}  R={:.4f}  F1={:.4f}".format(
                clf_name, acc, prec, rec, f1))

    return results


def cross_validate(texts, labels):
    """5-fold CV on TF-IDF features for each classifier."""
    print("\n--- 5-Fold Cross-Validation (TF-IDF) ---")
    vec = TfidfVectorizer(lowercase=True, stop_words="english", max_features=10000)
    X = vec.fit_transform(texts)
    y = np.array(labels)
    cv = {}
    for name, clf in get_classifiers().items():
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        cv[name] = (scores.mean(), scores.std())
        print("  {:<15s} {:.4f} +/- {:.4f}".format(name, scores.mean(), scores.std()))
    return cv

# ─────────────────────────────────────────────
#  6. Result visualizations
# ─────────────────────────────────────────────

def plot_comparison(results, out_dir):
    """Grouped bar chart comparing all configurations."""
    feat_names = ["BoW", "TF-IDF", "Bigram"]
    clf_names  = [c for c in get_classifiers()]
    colors     = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    metrics    = ["acc", "prec", "rec", "f1"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Performance Comparison (BBC Sport vs Politics)", fontsize=13, fontweight="bold")
    bw = 0.18
    x  = np.arange(len(feat_names))

    for mi, metric in enumerate(metrics):
        ax = axes[mi//2][mi%2]
        for ci, cn in enumerate(clf_names):
            vals = [next(r[metric] for r in results
                         if r["feature"]==fn and r["clf"]==cn)
                    for fn in feat_names]
            ax.bar(x + ci*bw, vals, bw, label=cn, color=colors[ci])
        ax.set_xticks(x + 1.5*bw)
        ax.set_xticklabels(feat_names)
        ax.set_ylim(0.85, 1.02)
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout(rect=[0,0,1,0.95])
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved", path)


def plot_confusion_matrices(results, out_dir):
    """One confusion matrix per classifier (best feature config)."""
    clf_names = list(get_classifiers().keys())
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    fig.suptitle("Confusion Matrices (best feature per classifier)", fontsize=11, fontweight="bold")
    for ci, cn in enumerate(clf_names):
        best = max((r for r in results if r["clf"]==cn), key=lambda r: r["f1"])
        cm = best["cm"]
        ax = axes[ci]
        ax.imshow(cm, cmap="Blues")
        ax.set_title("{}\n({})".format(cn, best["feature"]), fontsize=9)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(NAMES); ax.set_yticklabels(NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i][j]), ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i][j] > cm.max()/2 else "black")
    plt.tight_layout()
    path = os.path.join(out_dir, "confusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved", path)


def plot_cv(cv_results, out_dir):
    """Bar chart of cross-validation accuracies."""
    names = list(cv_results.keys())
    means = [cv_results[n][0] for n in names]
    stds  = [cv_results[n][1] for n in names]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors)
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.004,
                "{:.3f}".format(m), ha="center", fontsize=10)
    ax.set_ylabel("Accuracy")
    ax.set_title("5-Fold CV Accuracy (TF-IDF)")
    ax.set_ylim(0.90, 1.02)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "cv.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("  Saved", path)

# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 55)
    print("  Sports vs Politics Classifier - b22ch045")
    print("=" * 55)

    # 1. data
    print("\n[1] Downloading BBC News dataset...")
    bbc_root = download_bbc()
    texts, labels = load_articles(bbc_root)
    print("  Loaded {} documents.".format(len(texts)))
    print_stats(texts, labels)

    # 2. dataset plots
    print("\n[2] Dataset visualizations...")
    plot_class_distribution(labels, out_dir)
    plot_top_words(texts, labels, out_dir)

    # 3. train/test split (80/20, stratified)
    print("\n[3] Splitting 80/20...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels)
    print("  Train: {}  Test: {}".format(len(X_tr), len(X_te)))

    # 4. experiments
    print("\n[4] Training & evaluating (4 classifiers x 3 features)...")
    results = run_experiments(X_tr, X_te, y_tr, y_te)

    # 5. cross-validation
    print("\n[5] Cross-validation...")
    cv = cross_validate(texts, labels)

    # 6. result plots
    print("\n[6] Result visualizations...")
    plot_comparison(results, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_cv(cv, out_dir)

    # summary table
    print("\n" + "=" * 68)
    print("{:<35s} {:>7s} {:>7s} {:>7s} {:>7s}".format(
        "Configuration", "Acc", "Prec", "Rec", "F1"))
    print("-" * 68)
    for r in results:
        print("{:<35s} {:>7.4f} {:>7.4f} {:>7.4f} {:>7.4f}".format(
            r["label"], r["acc"], r["prec"], r["rec"], r["f1"]))

    best = max(results, key=lambda r: r["f1"])
    print("\nBest: {} (F1 = {:.4f})".format(best["label"], best["f1"]))
    print("Done!")


if __name__ == "__main__":
    main()
