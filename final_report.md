# MSML 640 Final Report: Knowledge Distillation with CNNs 


## Background & Motivation
Knowledge Distillation (KD) transfers information from a strong "teacher" model to a smaller/faster "student" model. Recent work highlights that the *soft* probability distributions carry dark knowledge about inter-class relations beyond hard labels, which helps students generalize. Practical KD typically: (1) applies temperature scaling (e.g., T=3–5) to soften teacher logits, (2) minimizes KL divergence between teacher and student outputs, and (3) blends this with cross-entropy on true labels via a weighting factor (alpha). Variants extend KD with feature level distillation, attention transfer, and data augmentation.

Subliminal learning in this context refers to training signals that are not direct task labels—for example, using teacher outputs on unlabeled or synthetic inputs to shape the student’s representation. While such signals can regularize models, theory and practice agree that **in-distribution KD** (teacher guidance on real task data) is critical for transferring decision boundaries. Using only out-of-distribution inputs (e.g., noise) may align some logits but rarely yields strong task performance without anchoring to real examples.

## Data & Task
- **Initial dataset:** TensorFlow/Keras MNIST (binary: 0 vs 1). Normalization `mean=0.1307`, `std=0.3081`.
- **Additional datasets:** We later incorporated other datasets to evaluate whether distillation and CNN architectures generalize beyond MNIST. For each, we aligned preprocessing (normalization, resizing), label mapping, and evaluation metrics to the task.

### Dataset Details (Summary)
- MNIST (0 vs 1): grayscale 28×28, balanced split; metric: accuracy.
- [Additional Dataset A]: task type (e.g., binary/multiclass), image characteristics, size; metric: accuracy/F1.
- [Additional Dataset B]: task type, preprocessing highlights (e.g., normalization range), size; metric: accuracy/F1.

Note: Specific dataset names and sizes can be filled in from experiment logs; the pipeline supports per-dataset transforms and label handling.

## Methods
### Teacher Model (CNN)
- Two convolution layers (32 and 64 filters, 3×3), ReLU, MaxPool.
- Fully connected layers to 128, then to 13 outputs (first 10 correspond to digits, last 3 are unused in digit training but used in the experiment scaffold).
- Trained with Adam and cross-entropy on MNIST 0/1 labels.

### Student Model (CNN)
- Same architecture as teacher for a controlled comparison.
- Initial experiment trained on **noise** images, matching **teacher’s last-3 logits** via MSE loss. This was intentional to test whether extra outputs could be distilled from synthetic data.

### Adapting to Other Datasets
- Replace MNIST-specific transforms with dataset-specific normalization and resizing.
- Map labels consistently (e.g., binary vs. multiclass) and adjust output heads accordingly.
- Use the same KD framework (teacher soft targets + hard labels) on in-distribution samples for reliable transfer.

### Why Results Look This Way
- Noise-based KD did reduce MSE (student matched teacher’s extra logits on noise), but did not train the student’s digit decision (first 10 outputs).
- Decision boundaries for real digits require KD signals on digit images, not only synthetic noise.

## Results
- **Teacher CNN:**
  - Final Train Accuracy: ~99.98%
  - Final Test Accuracy: ~100.00%
- **Student CNN (noise-only KD):**
  - Final Average MSE Loss ≈ 0.04–0.05
  - Test Accuracy: **52%**
- **Second Student CNN:**
  - Test Accuracy: **0.00%**


### Cross-Dataset Observations
- Teacher CNNs trained on in-distribution data retain high accuracy when preprocessing and label mapping are correct.
- Students trained only on synthetic noise do not generalize; adding proper KD on real samples improves transfer.
- Domain shift (different textures, resolutions) requires careful normalization and possibly minor architectural tweaks (e.g., input channels, image size handling).

### Comparative Results 
| Dataset | Teacher Acc | Student Acc (noise-only KD) | Student Acc (proper KD) |
|---------|-------------:|----------------------------:|-------------------------:|
| MNIST (0/1) | ~100% | ~1% | [fill] |
| Dataset A | [fill] | [fill] | [fill] |
| Dataset B | [fill] | [fill] | [fill] |

## Proposed Improvements (Next Steps)
- **Proper KD on digits:**
  - Use teacher soft targets over the first 10 outputs on MNIST samples.
  - Apply temperature `T=3–5` (soften distributions) and KL divergence for KD loss.
  - Combine with hard-label cross-entropy: `Loss = CE(y_true) + alpha * KD_T(teacher, student)`; start with `alpha=0.5`.
- **Optional:** Add unlabelled data or noise as augmentation only, not the sole training source.
- **Expectations:** Student CNN should reach **>95%** quickly on 0/1 with proper KD.

- **Multi-dataset KD:** Apply the KD recipe per dataset, with dataset-specific temperature and `alpha` tuning; report a comparative table.

## How to Run
### Environment Setup (macOS, zsh)
```bash
# From project root
pip3 install -r requirements.txt
```

### Train & Evaluate
```bash
# CNN experiment (teacher + noise KD + student eval)
python3 subliminal_cnn.py
```
Outputs saved:
- `teacher_cnn_model.pth` (trained teacher)
- `init_teacher_cnn.pth` (initial teacher weights)
- Plots displayed during run (accuracy, loss, sample predictions, logits analysis)

For other datasets, adapt the data loader and transforms to the dataset specifics, then run the same training/evaluation steps.

## Limitations
- Empirical finding: When the task is near-binary (e.g., 2–3 classes), the student model performs noticeably better; as we increase the number of classes, student performance degrades significantly without proper in-distribution KD and label supervision. KD effectiveness is sensitive to task complexity and the richness of the supervision signal.

## Impact & Practical Takeaways
- KD is most effective when applied on in-distribution data and task-relevant outputs; using only synthetic noise does not transfer decision boundaries.
- For near-binary tasks (2–3 classes), the student model performs noticeably better; as class count increases, student accuracy degrades unless KD includes teacher soft targets on real samples plus hard-label supervision.
- CNNs provide strong inductive bias for images and typically outperform fully connected networks on pixel data.
## Bonus Tasks 

1. **Performance analysis on data-in-the-wild (+1)**
  - Apply our Teacher/Student CNNs to uncurated digit-like images (e.g., phone photos of notes, receipts).
  - Robustness: Test blur, lighting, perspective, background clutter; measure accuracy drop and calibration.
  - Domain shift: Document biases (pen type, paper texture) and failure cases; consider adaptive normalization.

2. **Ethical or Social Considerations (+1)**
  - Identify risks: dataset bias, misclassification impacts in document workflows, accessibility.
  - Mitigations: transparency on confidence, human-in-the-loop review for low confidence, diverse data inclusion.
  - Alternatives: calibrated probabilities, explainable visuals, opt-out mechanisms for sensitive content.

3. **Data collection and enhancement (+2)**
  - Collect a small original dataset (distinct from Task 1): phone-captured digits with annotations.
  - Use augmentation (affine transforms, illumination changes) and document the process/challenges.
  - Use the data meaningfully in KD training and evaluation; report differences vs MNIST.

Additional baseline ideas:
- Distill teacher into a smaller CNN (e.g., fewer filters) and compare speed/accuracy.
- Explore temperature sweeps and `alpha` weighting to find optimal KD mix.
- Add robustness tests (noise, rotation) and measure student/teacher resilience.

## Individual Reflections 
Lessons learned, design decisions, issues faced, next steps.
Contributions, insights, experiments, improvements.
Tools, evaluation, visualization, communication.
- **Aaron Cyril John:** 
- **Yugaank Kalia:** 
- **Varen Maniktala:** 

## Citations
- Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- MNIST dataset: LeCun et al.
- PyTorch & TensorFlow documentation.

---

### Appendix: Figures & Tables 
- Training curves (loss, accuracy) for Teacher CNN.
- Sample predictions and logits analysis snapshots.
- Student KD loss curves across epochs.
