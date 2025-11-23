# AI Query System - Defense Intelligence

## Overview
Advanced AI-powered query system that integrates Hugging Face models for natural language interaction with defense intelligence datasets.

## Features
- **Image Captioning**: BLIP model for automatic image descriptions
- **Visual Q&A**: ViLT model for answering questions about images
- **Dataset Registry**: Built-in catalog of defense datasets
- **Natural Language Queries**: Ask questions about datasets in plain English
- **Query History**: Automatic logging of all interactions
- **GPU Acceleration**: CUDA support when available

## Setup

### 1. Install Dependencies
```powershell
cd c:\Users\ramas\OneDrive\Desktop\Betaroot
python -m venv venv
venv\Scripts\activate
pip install transformers torch pillow
```

### 2. Set Hugging Face API Key
```powershell
set HF_API_KEY=your_hugging_face_api_key_here
```

### 3. Run the System
```powershell
python ai_query_system.py
```

## Usage

### Interactive Commands
- `dataset <name>` - Get details about a specific dataset
- `recommend <need>` - Get dataset recommendations (e.g., "recommend satellite imagery")
- `caption <image_path>` - Generate caption for an image
- `vqa <image_path>::<question>` - Ask a question about an image
- `history` - View query history
- `quit` - Exit the system

### Example Queries
```
>> list datasets
>> recommend satellite imagery
>> dataset signal_intel
>> caption visualizations/signal_dashboard.png
>> vqa satellite_image.jpg::What vehicles are visible?
```

## Available Datasets
- **satellite_targets**: Multispectral satellite imagery for military asset detection
- **signal_intel**: RF signal snapshots for SIGINT classification  
- **cyber_threats**: Network telemetry for cyber intrusion detection

## Model Details
- **BLIP**: Salesforce/blip-image-captioning-large
- **ViLT**: dandelin/vilt-b32-finetuned-vqa
- **GPU**: CUDA support when available
- **Fallback**: Mock responses when models fail to load

## Files Created
- `ai_query_system.py` - Main application
- `logs/ai_query_system.log` - Application logs
- `query_history.json` - Query history (auto-generated)

## Integration with Existing System
This AI query system complements the existing defense intelligence infrastructure:
- Uses same MongoDB Atlas connection
- Integrates with Tavily API for dataset discovery
- Compatible with existing visualization outputs
- Extends natural language capabilities to stored data
