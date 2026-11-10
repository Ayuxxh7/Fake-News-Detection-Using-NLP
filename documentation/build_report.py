"""
build_report.py
---------------
Builds documentation/Fake_News_Detection_Project_Report.pdf (3 pages) using the
ACTUAL results stored in model/metrics.json (created by train_model.py).

Run:  python documentation/build_report.py
Needs: reportlab, matplotlib
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

DOC_DIR = Path(__file__).resolve().parent
BASE_DIR = DOC_DIR.parent
METRICS_PATH = BASE_DIR / "model" / "metrics.json"
CM_IMAGE = DOC_DIR / "confusion_matrix.png"
ARCH_IMAGE = DOC_DIR / "architecture.png"
PDF_PATH = DOC_DIR / "Fake_News_Detection_Project_Report.pdf"

NAVY = colors.HexColor("#1a3c6e")


def _aspect(path):
    """height / width of an image file (so pictures are not stretched)."""
    from PIL import Image as PILImage
    with PILImage.open(path) as im:
        return im.height / im.width


def make_architecture_image(path):
    """Draw a simple block diagram of the system with matplotlib."""
    steps = ["User enters\nnews text", "Preprocessing\n(clean, tokenize,\nstopwords)", "TF-IDF\nVectorizer",
             "Logistic\nRegression", "REAL / FAKE\n+ model score"]
    fig, ax = plt.subplots(figsize=(9, 1.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 20)
    ax.axis("off")
    w, gap = 16.5, 4.25
    for i, label in enumerate(steps):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 3), w, 14, boxstyle="round,pad=0.4",
                                    fc="#e8eef8", ec="#1a3c6e", lw=1.4))
        ax.text(x + w / 2, 10, label, ha="center", va="center", fontsize=8.5, color="#1a3c6e")
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.3, 10), xytext=(x + w + 0.5, 10),
                        arrowprops=dict(arrowstyle="->", color="#1a3c6e", lw=1.5))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    if not METRICS_PATH.exists():
        raise SystemExit("model/metrics.json not found. Run 'python train_model.py' first.")
    m = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    cmat = m["confusion_matrix"]["matrix"]
    make_architecture_image(ARCH_IMAGE)

    ss = getSampleStyleSheet()
    title = ParagraphStyle("T", parent=ss["Title"], fontSize=19, textColor=NAVY, spaceAfter=2)
    subtitle = ParagraphStyle("Sub", parent=ss["Normal"], fontSize=9.5, alignment=TA_CENTER,
                              textColor=colors.grey, spaceAfter=8)
    h = ParagraphStyle("H", parent=ss["Heading2"], fontSize=12.5, textColor=NAVY, spaceBefore=6, spaceAfter=2)
    body = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=12.4, alignment=TA_JUSTIFY, spaceAfter=3)
    bullet = ParagraphStyle("Bu", parent=body, leftIndent=14, bulletIndent=3, spaceAfter=1.5)

    def P(t): return Paragraph(t, body)
    def H(t): return Paragraph(t, h)
    def B(items): return [Paragraph(i, bullet, bulletText="•") for i in items]

    def table(data, widths, header=True):
        t = Table(data, colWidths=widths)
        style = [("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9aa7bd")),
                 ("FONTSIZE", (0, 0), (-1, -1), 9),
                 ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                 ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
        if header:
            style += [("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                      ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
        t.setStyle(TableStyle(style))
        return t

    s = []

    # ---------------- PAGE 1 ----------------
    s += [Paragraph("Fake News Detection Using NLP", title),
          Paragraph("A Machine Learning Mini Project &nbsp;|&nbsp; Python, Scikit-learn, TF-IDF, Streamlit", subtitle),
          H("1. Introduction"),
          P("News spreads very fast on social media, and false or misleading stories spread along with it. "
            "Checking every story by hand is slow. Natural Language Processing (NLP) and Machine Learning let a "
            "computer learn patterns from examples of real and fake news and give a quick first judgement on a new text. "
            "This project builds a small, working system that does exactly that and shows the result in a simple web page."),
          H("2. Problem Statement"),
          P("Given a news headline or article written in English, automatically classify it as <b>REAL</b> or <b>FAKE</b> "
            "using only its text, and present the result through an easy-to-use web interface."),
          H("3. Objectives"),
          *B(["Clean raw news text using NLP preprocessing techniques.",
              "Convert text into numerical features using TF-IDF.",
              "Train a Logistic Regression classifier and evaluate it with Accuracy, Precision, Recall, F1 score and a Confusion Matrix.",
              "Build a Streamlit application where a user can paste news and get a prediction.",
              "Run smoothly on a normal college laptop with beginner-friendly code."]),
          H("4. Technologies Used"),
          table([["Technology", "Purpose"],
                 ["Python 3", "Main programming language"],
                 ["Pandas, NumPy", "Reading the dataset and handling data/arrays"],
                 ["Regular expressions (re)", "Text cleaning and tokenization"],
                 ["Scikit-learn", "TF-IDF vectorizer, Logistic Regression, evaluation metrics"],
                 ["Joblib", "Saving and loading the trained model"],
                 ["Matplotlib", "Confusion matrix and diagram images"],
                 ["Streamlit", "Web user interface"]],
                [5.2 * cm, 11.3 * cm]),
          Spacer(1, 6),
          H("5. Dataset Used"),
          P(f"The project uses a <b>small synthetic sample dataset</b> ({m['total_rows']} items: "
            f"{m['class_counts'].get('REAL', 0)} REAL-style and {m['class_counts'].get('FAKE', 0)} FAKE-style) "
            "created by the included script so that the project runs without any download. REAL-style items are written in a neutral "
            "reporting tone and FAKE-style items in a sensational, clickbait tone. It is <b>not real-world news</b>. "
            "The code also accepts any real CSV dataset with <i>text</i> and <i>label</i> columns."),
          PageBreak()]

    # ---------------- PAGE 2 ----------------
    s += [H("6. What is NLP?"),
          P("Natural Language Processing (NLP) is the branch of Artificial Intelligence that helps computers read, process and "
            "understand human language. Computers only understand numbers, so NLP converts text into a numerical form "
            "that Machine Learning algorithms can use."),
          H("7. How the Project Works (Step by Step)"),
          *B(["<b>Step 1 - Load data:</b> read <i>news.csv</i> with Pandas, check columns, remove empty rows and duplicates.",
              "<b>Step 2 - Preprocess:</b> clean every text (explained below).",
              f"<b>Step 3 - Split:</b> 80% training ({m['train_rows']} rows) and 20% testing ({m['test_rows']} rows), keeping both classes balanced.",
              "<b>Step 4 - Vectorize:</b> convert text to TF-IDF numbers (vocabulary learned from training data only).",
              "<b>Step 5 - Train:</b> fit a Logistic Regression model on the training numbers.",
              "<b>Step 6 - Evaluate:</b> predict on the unseen test set and compute the metrics.",
              "<b>Step 7 - Save and use:</b> save the vectorizer and model with Joblib; the Streamlit app loads them to predict new text."]),
          H("8. Text Preprocessing"),
          table([["Step", "What it does", "Example"],
                 ["Lowercase", "Makes 'NEWS' and 'news' the same word", "BREAKING -> breaking"],
                 ["Remove noise", "Removes URLs, HTML tags, punctuation, digits", "100%! -> (removed)"],
                 ["Tokenization", "Splits the sentence into words", "'hidden truth' -> [hidden, truth]"],
                 ["Stopword removal", "Removes common words with little meaning", "the, is, and, of -> removed"]],
                [3.3 * cm, 7.2 * cm, 6 * cm]),
          Spacer(1, 3),
          P("Example from the project: <i>'BREAKING!!! Visit http://fake.site NOW - The government has 100% hidden the truth!'</i> "
            "becomes <i>'breaking visit government hidden truth'</i>."),
          H("9. TF-IDF Vectorization"),
          P("TF-IDF (Term Frequency - Inverse Document Frequency) gives every word a weight. <b>TF</b> measures how often a word "
            "appears in a text; <b>IDF</b> lowers the weight of words that appear in almost every text and raises the weight of "
            "rarer, more informative words. Here single words and word pairs (unigrams + bigrams) are used, up to 5000 features."),
          H("10. Machine Learning Algorithm: Logistic Regression"),
          P("Logistic Regression is a simple, fast and easy-to-explain classification algorithm. It learns a weight for each TF-IDF "
            "feature, adds them up, and converts the sum into a probability between 0 and 1 (using the sigmoid function). "
            "If the probability of REAL is higher than that of FAKE, the text is labelled REAL; otherwise FAKE. "
            "It runs in seconds on a normal laptop."),
          H("11. Model Training and Prediction"),
          P("During training the model adjusts its weights to reduce mistakes on the training texts. During prediction, the app "
            "applies the <b>same</b> preprocessing and the <b>same</b> saved TF-IDF vocabulary to the user's text, passes the numbers to "
            "the model, and shows the label and the model's probability as a <i>model score</i> (not a verified confidence)."),
          PageBreak()]

    # ---------------- PAGE 3 ----------------
    results = [["Metric (positive class = REAL)", "Measured value"],
               ["Accuracy", f"{m['accuracy']:.2%}"],
               ["Precision", f"{m['precision']:.2%}"],
               ["Recall", f"{m['recall']:.2%}"],
               ["F1 score", f"{m['f1_score']:.2%}"],
               ["5-fold CV accuracy (training part)", f"{m['cv_accuracy_mean']:.2%} (+/- {m['cv_accuracy_std']:.2%})"],
               ["Brier score (probability quality, lower is better)", f"{m['brier_score']:.3f}"]]
    cm_table = table([["", "Predicted FAKE", "Predicted REAL"],
                      ["Actual FAKE", str(cmat[0][0]), str(cmat[0][1])],
                      ["Actual REAL", str(cmat[1][0]), str(cmat[1][1])]],
                     [2.6 * cm, 2.9 * cm, 2.9 * cm])
    cm_img = Image(str(CM_IMAGE), width=4.4 * cm, height=3.77 * cm)
    side = Table([[cm_table, cm_img]], colWidths=[9.2 * cm, 7.3 * cm])
    side.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))

    s += [H("12. System Architecture"),
          Image(str(ARCH_IMAGE), width=15.5 * cm, height=15.5 * cm * _aspect(ARCH_IMAGE)),
          H("13. Model Evaluation Metrics and Results"),
          P("The metrics were measured by running <i>train_model.py</i> on the 20% test split "
            f"({m['test_rows']} items never seen during training). <b>Accuracy</b> = correct predictions / all predictions. "
            "<b>Precision</b> = of the texts predicted REAL, how many were really REAL. <b>Recall</b> = of the really REAL texts, how many were found. "
            "<b>F1</b> = balance of precision and recall. The <b>confusion matrix</b> counts correct and wrong predictions per class."),
          table(results, [10 * cm, 6.5 * cm]),
          Spacer(1, 4), side,
          Spacer(1, 3),
          P("<b>Important note:</b> these scores are perfect because the sample dataset is small and synthetic and the test items share "
            "phrasing with the training items; they do <b>not</b> show real-world accuracy. On six hand-written articles not in the dataset "
            "the model labelled all 6 as expected, but a false claim written in calm, official language was wrongly labelled REAL, "
            "showing that the model reads writing style, not facts. Results on a real dataset must be measured separately."),
          KeepTogether([H("14. Advantages and Limitations"),
                        *B(["<b>Advantages:</b> simple, fast, runs offline on a normal laptop, easy to explain, easy to retrain with new data, "
                            "friendly web interface with error handling.",
                            "<b>Limitations:</b> learns words and style rather than facts; cannot verify claims or sources; sample dataset is synthetic; "
                            "English only; the model score is not a calibrated confidence; needs retraining as topics change."])]),
          KeepTogether([H("15. Future Scope"),
                        P("Train on a large real-world dataset; try other models (Passive Aggressive Classifier, SVM) and compare; "
                          "use deep learning models such as BERT; add support for Hindi and other languages; add fact-checking and source "
                          "credibility checks; calibrate the probabilities; deploy the app online.")]),
          KeepTogether([H("16. Conclusion"),
                        P("The project shows the full machine learning workflow for text: preprocessing, TF-IDF, training, evaluation, saving the "
                          "model and building a web app. It works as a learning demonstration; for real use it must be trained on real news data "
                          "and combined with human fact-checking.")])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawCentredString(A4[0] / 2, 1.0 * cm, f"Fake News Detection Using NLP  |  Page {doc.page}")
        canvas.restoreState()

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.6 * cm,
                            title="Fake News Detection Using NLP - Project Report")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print("PDF saved:", PDF_PATH)


if __name__ == "__main__":
    main()
