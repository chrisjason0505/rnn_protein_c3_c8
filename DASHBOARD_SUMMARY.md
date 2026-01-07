# 🧬 Protein Secondary Structure Prediction - TrackIO Dashboard

## 📊 Training Results Summary

### Q8 Model (8-class Secondary Structure)

**Final Training Statistics:**
- **Total Epochs:** 20
- **Best Validation F1:** 0.3406 (achieved at epoch 10)
- **Final Training Loss:** 0.8178
- **Final Validation F1:** 0.3362

**Training Progress:**
- Epochs 1-4: Initial training with LR=0.001
- Epochs 5-7: Continued with LR=0.001
- Epochs 8-12: Learning rate reduced to 0.0005 (best F1 achieved at epoch 10)
- Epochs 13-15: Learning rate reduced to 0.00025
- Epochs 16-18: Learning rate reduced to 0.000125
- Epochs 19-20: Learning rate reduced to 0.000063

**Key Milestones:**
- Epoch 4: F1 = 0.3312 (first major improvement)
- Epoch 5: F1 = 0.3314 (new best) 
- Epoch 9: F1 = 0.3381 (new best)
- Epoch 10: F1 = 0.3406 ⭐ **BEST MODEL**

---

## 🌐 Live Dashboard

**Your TrackIO Dashboard URL:**
```
https://huggingface.co/spaces/srees0101/protein-sst-tracking
```

**What's in the Dashboard:**
- 📈 Interactive training plots (Loss vs F1 over 20 epochs)
- 📊 Summary cards with key metrics
- 🌟 Gold star marking the best model (epoch 10)
- 📋 Complete training data table

---

## 📁 Project Files

**Main Files:**
- `notesree_trackio.ipynb` - Training notebook with TrackIO integration
- `app.py` - Gradio dashboard application
- `requirements.txt` - Dashboard dependencies
- `.trackio/25-t3-nppe2.db` - TrackIO database with all training metrics
- `submission.csv` - Final predictions

---

## 🎯 Model Architecture

**BiLSTM + CNN Hybrid:**
- Embedding: 22 vocab → 128 dims
- CNN: [64, 128] filters with kernels [3, 5]
- BiLSTM: 256 hidden units, 2 layers
- Output: 8 classes (H, E, C, G, I, B, T, S)

---

## ✅ Deliverables

1. ✅ Trained model with F1 = 0.3406
2. ✅ TrackIO experiment tracking (20 epochs logged)
3. ✅ HuggingFace Space dashboard (shareable URL)
4. ✅ Submission file generated

---

*Dashboard updated: 2025-12-16*
*Best F1 Score: 0.3406 at Epoch 10*
