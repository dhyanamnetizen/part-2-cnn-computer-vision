# Part 2: Computer Vision Problem Formulation and CNN Prototype

This milestone focuses on designing and implementing an automated surface defect classification network to optimize assembly line product quality using spatial feature maps.

## Problem Formulation & Corporate Value Mapping
* **Core ML Classification Task:** This is formulated as a **Balanced Multi-Class Image Classification problem**. The system maps inputs into one of four mutually exclusive target categories: `normal`, `scratch`, `dent`, or `stain`.
* **Industrial Context Use Case:** This network functions as the core analytical engine for an automated visual inspection module placed directly inside a manufacturing assembly line (e.g., automotive panels or high-end smartphone bodies). 
* **Business KPI Value Impacts:** * Replacing manual inspectors lowers human oversight errors and cuts operational blockages.
  * Caught defects prevent damaged materials from moving downstream, protecting final assembly tools and lowering material wastage rates.

---

## Dataset Profile & Spatial Insights
* **Total Image Footprint:** 480 independent surface snapshots.
* **Class Matrix Layout:** Exactly 120 images per target category, providing a perfectly balanced spatial landscape.
* **Dimensional Profiles:** Native structural frames are standard uniform dimensions. For model ingest, frames are scaled down to uniform `64x64x3` tensors to balance computational workload with feature edge preservation.
* **Observed Visual Attributes:** Categorized textures show specific geometric characteristics. `scratch` labels present sharp directional line gradients, `dent` frames show contrasting shadow drops, and `stain` attributes indicate circular, localized color distribution anomalies.

---

## Image Preprocessing Pipelines
1. **Intensity Scaling:** Scaled raw 8-bit color arrays (`0-255`) down to continuous `[0.0, 1.0]` floating-point values to ensure clean gradient steps during backpropagation.
2. **Dimension Standardization:** Standardized spatial shapes down to `64x64` pixel arrays to maintain fixed boundaries for the initial convolutional filters.
3. **Validation Setup Split:** Employed an 80/20 stratified configuration to separate the inputs into 384 training images and 96 evaluation images, keeping a balanced class footprint across both splits.

---

## CNN Layer-by-Layer Architectural Design

The network relies on a sequential feed-forward layout designed to extract features from coarse to fine details:

* **Input Sensor Layer:** Expects fixed tensor sizes matching shapes of `(64, 64, 3)`.
* **Conv2D Block 1:** Runs 32 independent filters (`3x3` kernels) across the frame to extract localized edge and ridge markers. It uses `ReLU` activation to introduce spatial non-linearity.
* **MaxPooling2D Block 1:** Uses a `2x2` stride matrix to downsample resolution by half, decreasing spatial size while retaining key max-activation attributes to prevent position shifting.
* **Conv2D Block 2:** Runs 64 deeper filters (`3x3` kernels) to combine basic edge layers into complex spatial structures like textures, deep corners, and blotch shapes.
* **MaxPooling2D Block 2:** Reduces spatial dimensions down to `16x16` maps to keep the parameter footprint small before passing arrays to the fully connected steps.
* **Flattening Matrix Layer:** Transforms the multi-dimensional feature array into a single 1D feature array of parameters.
* **Dense Connection Layer:** Uses 64 hidden nodes with `ReLU` activations to weigh cross-feature spatial correlations.
* **Output Classification Layer:** Uses 4 distinct output nodes mapped with a `Softmax` distribution to output a final array of target classification probabilities summing to 1.0.

---

## Core Technical Concepts & Operational Explanations

### 1. Convolution Operations vs. Standard Dense Layers
Standard dense networks read images as flattened, unstructured strings of pixel points, completely losing spatial context like neighborhood groupings and boundaries. In contrast, convolution layers use specialized filter kernels that slide across image coordinates. This lets them capture spatial patterns and match specific structural signatures regardless of where they appear on the surface.

### 2. Max Pooling Function Tasks
Max pooling steps downsample feature dimensions by scanning pixel blocks and keeping only the strongest value. This serves two key technical goals: it reduces the computational load by cutting data size for downstream layers, and it builds positional invariance, ensuring the model identifies a surface scratch whether it occurs in the upper-left corner or the center of the frame.

### 3. Softmax Activation Output Requirements
While hidden layers use ReLU to handle non-linear boundaries, the final classification step requires a Softmax activation. It converts raw, unscaled network outputs into an actual probability distribution where each value is bounded between 0 and 1, and all values sum to exactly 1.0. This gives the plant operator a clear confidence metric for automated sorting decisions.