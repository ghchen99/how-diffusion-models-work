# How Diffusion Models Work

A hands-on educational project implementing Denoising Diffusion Probabilistic Models (DDPM) and Denoising Diffusion Implicit Models (DDIM) for image generation. This repository contains four progressive laboratories that teach the fundamentals of diffusion models through practical implementation.

## 🎯 Overview

This project demonstrates how diffusion models work by implementing them from scratch using PyTorch. You'll learn to generate 16x16 pixel sprite images through a step-by-step approach, covering sampling, training, contextual generation, and fast sampling techniques.

## 📚 Laboratory Structure

### Lab 1: Sampling (`L1 Sampling/`)
**Goal**: Understand the basic sampling process of diffusion models
- Load a pre-trained diffusion model
- Implement DDPM sampling algorithm
- Generate images from pure noise
- Compare correct vs incorrect sampling (with/without noise injection)
- Visualize the denoising process step-by-step

**Key Learning**: How diffusion models reverse the noise process to generate images

### Lab 2: Training (`L2 Training/`)
**Goal**: Train a diffusion model from scratch
- Implement the complete training loop
- Learn the noise perturbation process
- Understand the loss function (MSE between predicted and actual noise)
- Observe model improvement across training epochs
- Compare results from different training stages (epochs 0, 4, 8, 31)

**Key Learning**: How diffusion models learn to predict and remove noise

### Lab 3: Context (`L3 Context/`)
**Goal**: Add conditional generation capabilities
- Extend the model to accept context vectors
- Train with sprite labels (human, non-human, food, spell, side-facing)
- Implement context masking during training
- Generate images with specific attributes
- Experiment with mixed contexts

**Key Learning**: How to control what diffusion models generate

### Lab 4: Fast Sampling (`L4 Fast Sampling/`)
**Goal**: Implement efficient sampling techniques
- Compare DDPM vs DDIM sampling methods
- Reduce sampling steps from 500 to 25 while maintaining quality
- Measure and compare sampling speeds
- Apply fast sampling to both unconditional and conditional models

**Key Learning**: How to make diffusion models practical for real-time applications

## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

**Dependencies:**
- torch
- torchvision
- tqdm
- matplotlib
- numpy
- ipython
- pillow

### Quick Start
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd how-diffusion-models-work
   ```

2. **Run Lab 1 (Sampling)**
   ```bash
   cd "L1 Sampling"
   jupyter notebook L1_Sampling.ipynb
   ```

3. **Progress through the labs sequentially**
   - Each lab builds upon the previous one
   - Pre-trained weights are provided to skip lengthy training

## 🏗️ Architecture

### Model Architecture
- **ContextUnet**: U-Net architecture with context embedding
- **Input**: 16×16 RGB images (3 channels)
- **Features**: 64 hidden dimensions
- **Context**: 5-dimensional vectors for conditional generation
- **Timesteps**: 500 diffusion steps

### Key Components
- **ResidualConvBlock**: Building block with skip connections
- **UnetDown/UnetUp**: Encoder/decoder blocks with skip connections
- **EmbedFC**: Embedding layers for timestep and context
- **CustomDataset**: Sprite dataset loader with label support

## 📊 Dataset

The project uses a custom sprite dataset:
- **Images**: 1,788 16×16 pixel sprites
- **Labels**: 5 categories (human, non-human, food, spell, side-facing)
- **Format**: NumPy arrays for efficient loading
- **Source**: Sprites by ElvGames, FrootsnVeggies, and kyrise

## 🔬 Key Concepts Demonstrated

### Diffusion Process
- **Forward Process**: Gradually add noise to real images
- **Reverse Process**: Learn to remove noise step by step
- **Noise Schedule**: Linear interpolation between β₁=1e-4 and β₂=0.02

### Training Techniques
- **Noise Prediction**: Model learns to predict added noise
- **Random Timesteps**: Sample different noise levels during training
- **Context Masking**: Randomly mask context during training for robustness

### Sampling Methods
- **DDPM**: Full 500-step sampling with stochastic elements
- **DDIM**: Deterministic sampling with fewer steps
- **Context Control**: Guide generation with semantic labels

## 📈 Results

- **Training Progress**: Visual comparison across epochs showing quality improvement
- **Context Control**: Generate specific sprite types on demand
- **Speed Comparison**: DDIM achieves ~20x speedup over DDPM
- **Quality**: High-quality 16×16 sprites with recognizable features

## 🛠️ Usage Examples

### Basic Sampling
```python
# Load pre-trained model
nn_model.load_state_dict(torch.load("weights/model_trained.pth"))

# Generate 32 samples
samples, intermediate = sample_ddpm(32)
```

### Context-Controlled Generation
```python
# Define context (human=1, food=0.6)
ctx = torch.tensor([[1,0,0.6,0,0]]).float()

# Generate with context
samples, _ = sample_ddpm_context(1, ctx)
```

### Fast Sampling
```python
# Generate quickly with DDIM (25 steps instead of 500)
samples, intermediate = sample_ddim(32, n=25)
```

## 📝 Educational Notes

### Important Training Considerations
- **CPU Training Warning**: Training is computationally intensive and may take hours on CPU
- **Pre-trained Weights**: Provided to enable immediate experimentation
- **Progressive Learning**: Each lab builds conceptual understanding

### Implementation Details
- **Noise Addition**: `sqrt(αₜ) * x + sqrt(1-αₜ) * noise`
- **Denoising**: Careful balance between noise removal and maintaining stochasticity
- **Context Embedding**: Learn joint representations of time and context

## 🙏 Acknowledgments

- **Sprites**: ElvGames, [FrootsnVeggies](https://zrghr.itch.io/froots-and-veggies-culinary-pixels), and [kyrise](https://kyrise.itch.io/)
- **Base Implementation**: Modified from [minDiffusion](https://github.com/cloneofsimo/minDiffusion)
- **Theory**: Based on [DDPM](https://arxiv.org/abs/2006.11239) and [DDIM](https://arxiv.org/abs/2010.02502) papers

## 📚 Further Reading

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970)

## 🤝 Contributing

This is an educational project. Feel free to:
- Report issues with the notebooks
- Suggest improvements to explanations
- Share interesting results or modifications
- Contribute additional visualization tools

---

**Note**: This project is designed for educational purposes to understand diffusion model fundamentals. For production applications, consider using established frameworks like Hugging Face Diffusers.