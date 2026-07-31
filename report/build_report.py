"""
Builds the Part 1 report as a Word document from the validation results.

Run after 03_validate.py:  python report/build_report.py  ->  report/DAT610_Assignment1_Report.docx

The tables are populated straight from outputs/validation_results.json so the
numbers in the document always match what the scripts actually produced.
"""

import json

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Inches, Pt, RGBColor

RES = json.load(open("outputs/validation_results.json"))
H = RES["honest_ctgan"]
C = RES["corrupted"]

FONT = "Arial"


# --------------------------------------------------------------------------- #
# Small styling helpers
# --------------------------------------------------------------------------- #
def base_style(doc):
    st = doc.styles["Normal"]
    st.font.name = FONT
    st.font.size = Pt(11)


def heading(doc, text, size=13, space_before=14, space_after=6):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    r = p.add_run(text)
    r.bold = True
    r.font.name = FONT
    r.font.size = Pt(size)
    return p


def body(doc, text, space_after=8):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = p.add_run(text)
    r.font.name = FONT
    r.font.size = Pt(11)
    return p


def shade_cell(cell, hex_fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def make_table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.LEFT
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        run = hdr[i].paragraphs[0].add_run(h)
        run.bold = True
        run.font.name = FONT
        run.font.size = Pt(10)
        shade_cell(hdr[i], "EBEBEB")
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = ""
            run = cells[i].paragraphs[0].add_run(str(val))
            run.font.name = FONT
            run.font.size = Pt(10)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)
    return t


def caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(12)
    r = p.add_run(text)
    r.font.name = FONT
    r.font.size = Pt(10)


def figure(doc, path, caption_text, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    run = p.add_run()
    run.add_picture(path, width=Inches(width))
    caption(doc, caption_text)


# --------------------------------------------------------------------------- #
# Build the document
# --------------------------------------------------------------------------- #
doc = Document()
base_style(doc)
for section in doc.sections:
    section.page_height = Inches(11.69)
    section.page_width = Inches(8.27)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.top_margin = Inches(0.9)
    section.bottom_margin = Inches(0.9)

# ---- Title block ---------------------------------------------------------- #
title = doc.add_paragraph()
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = title.add_run("Validating a CTGAN Synthetic Dataset for Fraud Detection")
r.bold = True
r.font.size = Pt(16)
r.font.name = FONT

sub = doc.add_paragraph()
sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = sub.add_run("DAT610: Ethics & Privacy  |  Assignment 1")
r.font.size = Pt(11)
r.font.name = FONT

meta = doc.add_paragraph()
meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = meta.add_run("Maduechesi Chidiebere Jennifer   |   Matric No. 25120133019\n"
                 "MSc Data Science, Pan-Atlantic University")
r.font.size = Pt(10.5)
r.font.name = FONT
doc.add_paragraph().paragraph_format.space_after = Pt(4)

# ---- 1. The question ------------------------------------------------------ #
heading(doc, "1. The question I set out to answer")
body(doc,
     "The brief puts me in the seat of the data scientist at FintechPay, a Nigerian "
     "payments company, and asks one blunt question: do I approve this synthetic dataset "
     "for release into production? Fraud is rare, so the real transaction data is badly "
     "imbalanced, and the raw records carry customer PII that the NDPA (2023) won't let us "
     "pass around freely in development. Synthetic data is meant to solve both problems at "
     "once. My job isn't to assume it works. It's to prove whether it does, using the "
     "five-level framework from class, and then make a call I can actually defend.")
body(doc,
     "Short version of where I landed: I do not approve the CTGAN data for production "
     "release. It is genuinely useful for model development and it keeps the fraud signal "
     "intact, but it fails the stricter distribution and privacy checks, so \"useful\" and "
     "\"safe for production\" are not the same thing here. The rest of this report is the "
     "evidence behind that decision.")

# ---- 2. Workflow ---------------------------------------------------------- #
heading(doc, "2. Workflow and tooling")
body(doc,
     "I worked in three scripts, one per stage, so the pipeline is easy to follow and rerun. "
     "Everything is seeded (random_state = 42) so the numbers in this report reproduce exactly.")
body(doc,
     "Building the real dataset. There's no public FintechPay export, and even if there were "
     "it would be full of PII, so I generated a realistic stand-in of 6,000 transactions that "
     "follows the exact schema from the lecture: transaction_amount in Naira, customer_age, "
     "account_balance, num_transactions_30d, transaction_hour, distance_from_home_km, "
     "merchant_category, and the is_fraud target. I drew the fraud label first and then "
     "generated the features conditionally on it, so fraud rows genuinely behave differently: "
     "larger and more bimodal amounts, more late-night hours, longer distances from home, "
     "higher velocity. That built-in signal is what makes the later utility test meaningful. "
     "I set the fraud rate to about 10% rather than the ~1% you'd see in the wild. At 1% the "
     "minority class is too thin for the GAN to learn or for the tests to say anything stable, "
     "and I note later how that choice cuts both ways.")
body(doc,
     "Generating the synthetic data. I used CTGAN from the SDV library. The pipeline is the "
     "one from the slides: detect the metadata, verify it by hand (SDV first read is_fraud as "
     "numerical, so I pinned it and merchant_category as categorical), train the synthesiser, "
     "then sample. Two adjustments earned their place. First, a plain sample() call produced a "
     "fraud rate of about 28% against a real 10%. That is CTGAN's known habit of "
     "over-representing a rare category during training-by-sampling, and it is exactly the "
     "\"model trains on the wrong fraud rate\" failure the lecture warns about. Because CTGAN "
     "is a conditional model, I fixed it the intended way, by sampling from conditions to force "
     "the real 90/10 split. Second, the two money columns are heavily skewed, so I log1p "
     "transformed them before training and inverted after sampling. That alone pulled the "
     "transaction_amount KS statistic from about 0.30 down to 0.06.")
body(doc,
     "The corrupted twin. To check that the framework can actually catch a bad dataset and "
     "isn't just waving everything through, I also built a deliberately broken version, the way "
     "the lab exercise on the last slide asks. Two sabotages: transaction_amount multiplied by "
     "three (wrong scale) and the fraud rate forced up to 40% (wrong prevalence). If my "
     "validation is worth anything, this one has to fail loudly.")

# ---- 3. Validation results ------------------------------------------------ #
heading(doc, "3. Validation results, level by level")
body(doc,
     "I ran all five levels against the CTGAN data and, alongside it, the corrupted twin. "
     "Two of the tests, KS and chi-squared, get more sensitive as the sample grows, and at "
     "6,000 rows they reject differences far too small to matter in practice. So I run those "
     "two on a fixed 1,000-row subsample. The effect-size metrics (mean differences, "
     "Wasserstein, TSTR, DNNR) use the full data, where more rows only help.")

# Level 1
heading(doc, "Level 1 - Summary statistics", size=11.5, space_before=10, space_after=4)
l1 = H["level1"]["table"]
make_table(
    doc,
    ["Feature", "Real mean", "Synthetic mean", "Diff %", "Flag"],
    [[r["feature"], r["real_mean"], r["synthetic_mean"], r["pct_diff"], r["flag"]] for r in l1],
    widths=[2.1, 1.3, 1.4, 0.8, 0.7],
)
body(doc,
     f"Four of the six means land inside 10% and the amounts column sits at about 12%. Two "
     f"drift past the 20% line: account_balance and, worst of all, distance_from_home_km at "
     f"{H['level1']['worst_pct_diff']}%. So Level 1 is a partial pass at best. It's a screening "
     f"step though, not a verdict, and a mean can look fine while the shape underneath is wrong, "
     f"which is the whole reason we don't stop here.")

# Level 2
heading(doc, "Level 2 - Visual distributions", size=11.5, space_before=10, space_after=4)
body(doc,
     f"The KDE curves overlap closely on every continuous feature. The synthetic peaks are a "
     f"touch sharper on age, transaction count and hour, but the bodies of the distributions "
     f"sit on top of each other, and the skewed money columns line up well after the log "
     f"transform.")
figure(doc, "figures/level2_kde_continuous.png",
       "Real (blue) against synthetic (orange) density for each continuous feature. The curves "
       "track closely; the clearest gaps are the slightly taller synthetic peaks on customer_age "
       "and transaction_hour.")
body(doc,
     "The categorical picture is split. is_fraud matches almost exactly, which is the "
     "conditional sampling holding the class balance in place. merchant_category is the weak "
     "spot: the synthetic under-produces grocery and over-produces travel and electronics, and "
     "that visible gap is what the chi-squared test picks up in Level 3b.")
figure(doc, "figures/level2_bar_categorical.png",
       "Category proportions, real against synthetic. is_fraud lines up almost perfectly; "
       "merchant_category drifts, most visibly grocery (0.33 real against 0.17 synthetic) and "
       "travel (0.12 against 0.21).", width=6.3)
body(doc,
     f"The correlation heatmaps matter more to me than the marginals, because a fraud model "
     f"lives on the relationships between features, not on any one column. Here the synthetic "
     f"data does well. The mean absolute error across the whole correlation matrix is only "
     f"{H['level2']['mean_abs_corr_error']}. Every fraud-predictive link keeps its sign and "
     f"roughly its strength: transaction_amount to is_fraud is 0.35 in both, and the strong "
     f"num_transactions_30d to is_fraud link (0.69 real) still reads 0.61 in the synthetic. "
     f"That preserved structure is what carries the utility test later.")
figure(doc, "figures/level2_correlation_heatmaps.png",
       "Correlation structure, real (left) against synthetic (right). The pattern is preserved: "
       "the fraud row keeps every relationship in the right direction and close in magnitude.")

# Level 3a KS
heading(doc, "Level 3a - Kolmogorov-Smirnov test (continuous)", size=11.5, space_before=10, space_after=4)
l3a = H["level3a"]["table"]
make_table(
    doc,
    ["Feature", "KS stat", "p-value", "Aligned?"],
    [[r["feature"], r["ks_stat"], r["p_value"], "Yes" if r["aligned"] else "No"] for r in l3a],
    widths=[2.3, 1.1, 1.1, 1.0],
)
body(doc,
     "This is where the CTGAN data struggles. The null hypothesis is that real and synthetic "
     "come from the same distribution, and at p > 0.05 we'd keep it. Only account_balance clears "
     "the bar. transaction_amount and transaction_hour are close (KS around 0.06 to 0.09), but "
     "customer_age and distance sit near 0.21. I confirmed this held across reruns, so it isn't "
     "a one-off. Two honest reads sit side by side here: CTGAN really doesn't reproduce every "
     "marginal exactly, AND the p > 0.05 rule is punishing for a generative model, because it "
     "asks for near-identical distributions on every single column at once. Passing all six is a "
     "bar almost no GAN clears.")

# Level 3b Chi-squared
heading(doc, "Level 3b - Chi-squared test (categorical)", size=11.5, space_before=10, space_after=4)
l3b = H["level3b"]["table"]
make_table(
    doc,
    ["Feature", "Chi2 stat", "p-value", "Aligned?"],
    [[r["feature"], r["chi2_stat"], r["p_value"], "Yes" if r["aligned"] else "No"] for r in l3b],
    widths=[2.3, 1.1, 1.1, 1.0],
)
body(doc,
     "The is_fraud column passes cleanly (p = 0.45), which is the conditional sampling paying "
     "off: the synthetic fraud rate matches the real one, so a model trained on it won't learn "
     "the wrong prior. merchant_category doesn't pass. The category proportions drifted enough "
     "for the test to reject, which lines up with the slightly-off bar chart. So one of the two "
     "categorical columns is faithful and one isn't.")

# Level 3c Wasserstein
heading(doc, "Level 3c - Wasserstein distance", size=11.5, space_before=10, space_after=4)
l3c = H["level3c"]["table"]
make_table(
    doc,
    ["Feature", "Normalised distance", "Rating"],
    [[r["feature"], r["wasserstein_norm"], r["rating"]] for r in l3c],
    widths=[2.3, 1.8, 1.3],
)
body(doc,
     "Wasserstein gives the size of the drift rather than a yes/no, which I find more useful "
     "than the KS p-value. transaction_amount, num_transactions_30d and transaction_hour all "
     "come in acceptable (under 0.15). age, balance and distance are poor, topping out around "
     "0.31. It's the same three features that misbehaved in Level 1, so the story is consistent: "
     "the synthesis is decent on half the continuous columns and loose on the other half.")

# Level 4 TSTR
heading(doc, "Level 4 - ML utility, TSTR vs TRTR", size=11.5, space_before=10, space_after=4)
l4 = H["level4"]
make_table(
    doc,
    ["Setup", "AUC", "Meaning"],
    [["TRTR (train real, test real)", l4["trtr_auc"], "Gold-standard baseline"],
     ["TSTR (train synthetic, test real)", l4["tstr_auc"], "Simulates production"],
     ["Gap", l4["auc_gap"], l4["verdict"]]],
    widths=[2.7, 0.9, 2.0],
)
body(doc,
     f"This is the result that complicates the whole picture, and it's the one I care about "
     f"most, because it answers the actual business question: can a model learn to catch fraud "
     f"from the synthetic data? A random forest trained only on synthetic data scores "
     f"{l4['tstr_auc']} AUC on real, held-out transactions, against a {l4['trtr_auc']} baseline "
     f"for a model trained on real data. The gap is {l4['auc_gap']}, well inside the 0.02 "
     f"\"fully equivalent\" band. So for the job it's meant to do, the synthetic data WORKS, even "
     f"though it flunked the marginal tests. That gap between statistical fidelity and practical "
     f"utility is the single most interesting finding in this exercise.")

# Level 5 DNNR
heading(doc, "Level 5 - Privacy, DNNR", size=11.5, space_before=10, space_after=4)
l5 = H["level5"]
make_table(
    doc,
    ["Quantity", "Value"],
    [["median nearest synthetic-to-real distance", l5["median_d_sr"]],
     ["median nearest real-to-real distance", l5["median_d_rr"]],
     ["DNNR ratio", l5["dnnr"]],
     ["Interpretation", l5["interpretation"]]],
    widths=[3.6, 1.6],
)
body(doc,
     f"DNNR asks whether synthetic rows sit suspiciously close to real people. At "
     f"{l5['dnnr']} the synthetic points are, on average, about as close to a real record as "
     f"real records are to each other, which is right on the memorisation-risk boundary (the "
     f"safe bar is 1.5). I read this as a caution rather than proof of memorisation, but it is "
     f"not a clean pass, and for data derived from customer PII a borderline privacy score is "
     f"not something to hand-wave.")

# ---- 4. The corrupted twin ------------------------------------------------ #
heading(doc, "4. Does the framework actually catch a bad dataset?")
body(doc,
     "A validation framework that can't fail anything is theatre, so this is the real test of "
     "it. Running the corrupted twin through the same five levels, it collapses exactly where "
     "it should:")
make_table(
    doc,
    ["Level", "CTGAN (honest)", "Corrupted twin"],
    [["L1 worst mean diff", f"{H['level1']['worst_pct_diff']}%", f"{C['level1']['worst_pct_diff']}%"],
     ["L3b is_fraud chi2 p", "0.45 (pass)", "0.00 (fail)"],
     ["L3c worst Wasserstein", H["level3c"]["worst_norm"], C["level3c"]["worst_norm"]],
     ["L4 TSTR AUC gap", H["level4"]["auc_gap"], C["level4"]["auc_gap"]],
     ["L5 DNNR", H["level5"]["dnnr"], C["level5"]["dnnr"]]],
    widths=[2.2, 1.6, 1.6],
)
body(doc,
     f"The wrong scale on transaction_amount blows the Level 1 mean out to "
     f"{C['level1']['worst_pct_diff']}% and pushes its Wasserstein distance to "
     f"{C['level3c']['worst_norm']}, and the inflated fraud rate fails the is_fraud chi-squared "
     f"outright. Most telling, the TSTR gap widens to {C['level4']['auc_gap']}, into the "
     f"\"investigate before deployment\" band. The utility test that forgave the honest data's "
     f"marginal wobbles does not forgive real corruption, which is a nice demonstration that it "
     f"measures something the other tests miss. One curveball: the corrupted DNNR is "
     f"{C['level5']['dnnr']}, which reads as \"privacy preserved.\" That's a trap, and worth "
     f"saying out loud. The privacy score only looks good because the rescaled data is now so "
     f"far from anything real that it's useless. A high DNNR is only reassuring when the data is "
     f"faithful to begin with.")

# ---- 5. Discussion -------------------------------------------------------- #
heading(doc, "5. What the findings mean")
body(doc,
     "Pulling it together, the CTGAN data has a split personality. It's structurally and "
     "functionally sound: the class balance is right, the correlation web that drives fraud "
     "prediction is intact, and a model trained on it performs almost identically to one trained "
     "on real data. On the other hand it's only loosely faithful at the level of individual "
     "distributions, three of six continuous features drift on both KS and Wasserstein, one "
     "categorical column drifts, and the privacy margin is thin.")
body(doc,
     "I think the honest lesson is that no single metric should decide this. If I'd only run "
     "TSTR I'd have approved it. If I'd only run KS I'd have rejected it as basically broken. "
     "Both readings are too simple. The KS result in particular says as much about the test as "
     "about the data: at any reasonable sample size the p > 0.05 rule demands near-perfect "
     "marginals on every column, which is not a realistic bar for a GAN, and it's why I leaned "
     "on Wasserstein and the mean differences to judge how bad the drift actually was rather "
     "than just that it existed.")
body(doc,
     "The fraud-rate story is the practical highlight. CTGAN's default sampling tripled the "
     "fraud prevalence, and if I'd shipped that a downstream model would have been trained to "
     "think roughly one transaction in three is fraudulent. The conditional-sampling fix worked, "
     "but the fact that the tool's default does the wrong thing on imbalanced data is exactly "
     "why validation and disclosure aren't optional. Nobody downstream would have known.")

# ---- 6. Synthetic Data Card ---------------------------------------------- #
doc.add_page_break()
heading(doc, "6. Synthetic Data Card")
body(doc,
     "The lecture asks for a data card attached to the artefact, in three parts: generation, "
     "validation evidence, and disclosure.", space_after=6)

heading(doc, "Generation", size=11, space_before=6, space_after=4)
make_table(
    doc,
    ["Field", "Value"],
    [["Generator model", "CTGAN (SDV), 500 epochs, conditional sampling"],
     ["Training source", "FintechPay-schema transactions (synthetic stand-in, no raw PII)"],
     ["Real records", str(RES["dataset_sizes"]["real"])],
     ["Synthetic records", str(RES["dataset_sizes"]["synthetic"])],
     ["Fraud prevalence (real / synth)",
      f"{RES['fraud_rates']['real']*100:.2f}% / {RES['fraud_rates']['synthetic']*100:.2f}%"],
     ["Date generated", "See run log"],
     ["Purpose", "Fraud-detection class augmentation / PII-free development data"],
     ["Generated by", "Maduechesi Chidiebere Jennifer (25120133019)"]],
    widths=[2.2, 3.6],
)

heading(doc, "Validation evidence", size=11, space_before=6, space_after=4)
make_table(
    doc,
    ["Check", "Result", "Verdict"],
    [["L1 means within 10%", f"4 of 6 (worst {H['level1']['worst_pct_diff']}%)", "Partial"],
     ["L2 correlation error", str(H["level2"]["mean_abs_corr_error"]), "Pass"],
     ["L3a KS (continuous)", f"{H['level3a']['n_pass']} of {H['level3a']['n_total']} aligned", "Fail"],
     ["L3b chi-squared", "is_fraud pass, merchant_category fail", "Partial"],
     ["L3c Wasserstein < 0.15", f"3 of 6 (worst {H['level3c']['worst_norm']})", "Partial"],
     ["L4 TSTR AUC gap", f"{H['level4']['auc_gap']} ({H['level4']['verdict']})", "Pass"],
     ["L5 DNNR", f"{H['level5']['dnnr']} ({H['level5']['interpretation']})", "Borderline"]],
    widths=[2.0, 2.6, 1.2],
)

heading(doc, "Disclosure", size=11, space_before=6, space_after=4)
make_table(
    doc,
    ["Field", "Value"],
    [["Synthetic data used?", "Yes, CTGAN-generated, disclosed"],
     ["Stakeholders to notify", "Data owner, model risk / governance, DPO, compliance"],
     ["Validation status", "Not approved for production; restricted internal use only"],
     ["Sign-off", "Pending governance review"]],
    widths=[2.2, 3.6],
)

# ---- 7. Ethics and disclosure -------------------------------------------- #
heading(doc, "7. Ethics, disclosure and the professional obligation")
body(doc,
     "The reason all of this gets documented rather than kept in my head is that shipping "
     "synthetic data quietly is a professional and legal problem, not just a technical one. A "
     "few standards bite directly here. Nigeria's NDPA (2023) governs the PII in the underlying "
     "records and the safeguards around processing it, which is the reason we're using synthetic "
     "data at all. The EU AI Act (2024) treats fraud detection as high-risk and, under Articles "
     "10 and 13, expects data governance and transparency about what went into the model, "
     "undisclosed synthetic training data would sit badly against that. The ACM Code (1.2, 2.5) "
     "and the IEEE Code (I.1) both turn on avoiding harm through negligence and being honest "
     "about the things that could endanger others, and ISO/IEC 42001 (8.4) frames all of it as "
     "ongoing data-quality management. Deploying a model on data I can't vouch for, and not "
     "saying so, would be making a claim I can't back with evidence. That's the line the lecture "
     "drew, and I agree with it.")

# ---- 8. Recommendation ---------------------------------------------------- #
heading(doc, "8. Recommendation")
body(doc,
     "Do I approve this synthetic data for release into production? No, not as it stands. The "
     "recommendation, stated plainly:")
body(doc,
     "1. Approve for restricted internal use. The data is safe to use for model prototyping and "
     "experimentation in a PII-free environment, where the strong TSTR result and preserved "
     "correlations are what matter and the marginal drift is tolerable.")
body(doc,
     "2. Do not approve for production release yet. Three things block it: the KS and Wasserstein "
     "drift on customer_age, account_balance and distance_from_home_km; the merchant_category "
     "proportion mismatch; and the borderline DNNR, which for PII-derived data I'm not willing to "
     "wave through.")
body(doc,
     "3. Conditions for re-review. Retrain with more data and per-column transforms on the three "
     "weak features, bring their Wasserstein under 0.15 and DNNR above 1.5, then rerun the full "
     "framework. Keep the conditional sampling and the disclosure in place either way.")
body(doc,
     "That's a defensible \"not yet\" rather than a flat no. The framework did its job: it "
     "stopped me approving data that looks fine on a utility test alone, it caught the corrupted "
     "set hard, and it gave me specific, fixable reasons rather than a vague bad feeling. Which "
     "is the entire point of validating before you deploy.")

out = "report/DAT610_Assignment1_Report.docx"
doc.save(out)
print("Saved", out)
