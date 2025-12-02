# BLIP CAM:Self Hosted Live Image Captioning with Real-Time Video Stream 🎥

This repository implements real-time image captioning using the BLIP (Bootstrapped Language-Image Pretraining) model. The system captures live video from your webcam, generates descriptive captions for each frame, and displays them in real-time along with performance metrics.

## 🚀 Features

- **Real-Time Video Processing**: Seamless webcam feed capture and display with overlaid captions
- **State-of-the-Art Captioning**: Powered by Salesforce's BLIP image captioning model (blip-image-captioning-large)
- **Hardware Acceleration**: CUDA support for GPU-accelerated inference
- **Performance Monitoring**: Live display of:
  - Frame processing speed (FPS)
  - GPU memory usage
  - Processing latency
- **Optimized Architecture**: Multi-threaded design for smooth video streaming and caption generation

## 📋 Requirements

- Python 3.8+
- NVIDIA GPU (optional, for CUDA acceleration)
- Webcam

### Core Dependencies
```
opencv-python>=4.5.0
torch>=1.9.0
transformers>=4.21.0
Pillow>=8.0.0
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/zawawiAI/BLIP_CAM.git
cd BLIP_CAM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python BLIP_CAM.py
```

## 💡 Use Cases

- **Accessibility Tools**: Real-time scene description for visually impaired users
- **Content Analysis**: Automated video content understanding and tagging
- **Smart Conferencing**: Enhanced video calls with automatic scene descriptions
- **Educational Tools**: Visual learning assistance and scene comprehension
- **Security Systems**: Intelligent surveillance with scene description capabilities

## 🎮 Usage Controls

- Press `Q` to quit the application
- Press `S` to save the current frame with caption
- Press `P` to pause/resume caption generation

## 🔧 Configuration

The application can be customized through the following parameters in `config.py`:
- Frame processing resolution
- Caption update frequency
- GPU memory allocation
- Model confidence threshold
- Display preferences



## 🙏 Acknowledgments

- Salesforce for the BLIP model
- PyTorch team for the deep learning framework
- Hugging Face for the transformers library


    


## 👩‍💻 About Me

Hi, I’m **Jhansi Challa**, a passionate developer who enjoys building intelligent interfaces and AI-powered applications. I enjoy working with modern front-end stacks, experimenting with AI-driven tools, and solving real-world problems through code. 

Let's connect on [LinkedIn](linkedin.com/in/challajhansi) or reach out via email at: **jhansichalla@26gmail.com**

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


© 2025 Jhansi Challa
